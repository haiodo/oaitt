"""
OAITT — Open AI Transformer Transcriber.

Пул ASR worker'ов для параллельного инференса.
Каждый worker — отдельный OS-процесс (multiprocessing.Process) со своей моделью.
GIL не мешает — процессы работают по-настоящему параллельно.

Архитектура:
    Главный процесс (uvicorn) → кладёт задачу (audio bytes) в общую task_queue →
    → свободный worker-процесс берёт задачу, выполняет transcribe() на своей модели →
    → результат (dict) отправляется обратно через result pipe →
    → главный процесс десериализует результат и возвращает вызывающему.

    Каждый worker при старте сам создаёт и загружает модель (в своём процессе).
    Audio передаётся как bytes (numpy .tobytes() / frombuffer) чтобы избежать
    проблем с pickle для больших массивов.

Features:
    - Health monitoring: фоновый поток периодически проверяет is_alive() worker'ов.
    - Auto-restart: упавшие worker-процессы автоматически перезапускаются.
    - Transcribe timeout: ожидание результата ограничено по времени.
    - Task retry: если worker умер во время обработки задачи, задача переотправляется.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import multiprocessing
import os
import queue
import signal
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# Таймаут ожидания результата transcribe (секунды). 0 = без таймаута.
TRANSCRIBE_TIMEOUT = int(os.environ.get("ASR_TRANSCRIBE_TIMEOUT", "600"))

# Интервал проверки здоровья worker'ов (секунды)
HEALTH_CHECK_INTERVAL = int(os.environ.get("ASR_HEALTH_CHECK_INTERVAL", "5"))

# Максимальное количество попыток retry для одной задачи
MAX_TASK_RETRIES = int(os.environ.get("ASR_MAX_TASK_RETRIES", "2"))

# Таймаут ожидания загрузки модели в worker'е (секунды)
MODEL_LOAD_TIMEOUT = int(os.environ.get("ASR_MODEL_LOAD_TIMEOUT", "600"))

# Максимальное количество перезапусков одного worker-слота
MAX_WORKER_RESTARTS = int(os.environ.get("ASR_MAX_WORKER_RESTARTS", "10"))

# ---------------------------------------------------------------------------
# Сообщения между главным процессом и worker'ами
# ---------------------------------------------------------------------------

# Sentinel для остановки worker'а
_STOP = "STOP"


def _make_task(
    task_id: str,
    audio: np.ndarray,
    task: str,
    language: Optional[str],
    word_timestamps: bool,
    output: str,
    options: Optional[dict],
    retry_count: int = 0,
) -> dict:
    """Сериализует задачу в dict для передачи через Queue."""
    return {
        "task_id": task_id,
        "audio_bytes": audio.tobytes(),
        "audio_dtype": str(audio.dtype),
        "audio_shape": audio.shape,
        "task": task,
        "language": language,
        "word_timestamps": word_timestamps,
        "output": output,
        "options": options,
        "retry_count": retry_count,
    }


def _parse_audio(msg: dict) -> np.ndarray:
    """Восстанавливает numpy array из сериализованной задачи.

    Note: .copy() обязателен — np.frombuffer() возвращает read-only массив,
    а PyTorch требует writable tensor для torch.from_numpy().
    """
    return np.frombuffer(msg["audio_bytes"], dtype=msg["audio_dtype"]).reshape(
        msg["audio_shape"]
    ).copy()


# ---------------------------------------------------------------------------
# Worker-процесс
# ---------------------------------------------------------------------------


def _worker_main(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    ready_event: multiprocessing.Event,
):
    """
    Точка входа worker-процесса.

    Создаёт модель, загружает её, сигналит ready, крутит цикл обработки задач.
    """
    # Игнорируем SIGTERM/SIGINT в worker'ах — ими управляет главный процесс
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Настраиваем логирование в дочернем процессе
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [worker-{worker_id}] %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger(f"asr.worker.{worker_id}")
    log.info(f"Worker process {worker_id} starting (pid={os.getpid()})")

    try:
        # Импортируем и создаём модель внутри дочернего процесса
        from src.asr.factory import create_asr_model

        model = create_asr_model()
        model.ensure_model_loaded()
        log.info(f"Worker {worker_id} model loaded: {model.__class__.__name__}")
    except Exception as exc:
        log.error(f"Worker {worker_id} failed to load model: {exc}\n{traceback.format_exc()}")
        # Сигналим ready даже при ошибке, чтобы главный процесс не завис
        ready_event.set()
        return

    # Сигналим что готовы
    ready_event.set()

    # Основной цикл обработки задач
    while True:
        try:
            msg = task_queue.get()
        except Exception:
            break

        if msg is _STOP or msg == _STOP:
            log.info(f"Worker {worker_id} received stop signal")
            break

        task_id = msg.get("task_id", "?")
        log.info(f"Worker {worker_id} processing task {task_id}")
        start_time = time.perf_counter()

        try:
            audio = _parse_audio(msg)

            result = model.transcribe(
                audio=audio,
                task=msg["task"],
                language=msg["language"],
                word_timestamps=msg["word_timestamps"],
                output=msg["output"],
                options=msg["options"],
            )

            elapsed = time.perf_counter() - start_time
            log.info(f"Worker {worker_id} task {task_id} done in {elapsed:.2f}s")

            # Сериализуем результат
            if isinstance(result, str):
                result_data = {"type": "str", "value": result}
            else:
                # Pydantic model → dict
                result_data = {"type": "model", "value": result.model_dump()}

            result_queue.put({
                "task_id": task_id,
                "success": True,
                "result": result_data,
                "worker_id": worker_id,
                "elapsed": elapsed,
            })

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            log.error(
                f"Worker {worker_id} task {task_id} failed after {elapsed:.2f}s: "
                f"{exc}\n{traceback.format_exc()}"
            )
            result_queue.put({
                "task_id": task_id,
                "success": False,
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
                "worker_id": worker_id,
                "elapsed": elapsed,
            })

        # Помогаем GC
        del msg

    # Cleanup
    try:
        model.release_model()
        log.info(f"Worker {worker_id} model released")
    except Exception:
        pass
    log.info(f"Worker {worker_id} exiting")


# ---------------------------------------------------------------------------
# Dispatcher-поток: разбирает result_queue и будит вызывающих
# ---------------------------------------------------------------------------


class _ResultDispatcher(threading.Thread):
    """
    Фоновый поток в главном процессе.
    Читает результаты из result_queue и будит соответствующих вызывающих.
    """

    def __init__(self, result_queue: multiprocessing.Queue):
        super().__init__(daemon=True, name="asr-result-dispatcher")
        self._result_queue = result_queue
        self._waiters: dict[str, threading.Event] = {}
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = False

    def register(self, task_id: str) -> threading.Event:
        """Регистрирует ожидание результата. Возвращает Event для wait()."""
        event = threading.Event()
        with self._lock:
            self._waiters[task_id] = event
        return event

    def get_result(self, task_id: str) -> dict:
        """Забирает результат (вызывать после event.wait())."""
        with self._lock:
            return self._results.pop(task_id, {})

    def cancel(self, task_id: str) -> None:
        """Отменяет ожидание результата (при таймауте или retry)."""
        with self._lock:
            self._waiters.pop(task_id, None)
            self._results.pop(task_id, None)

    def has_result(self, task_id: str) -> bool:
        """Проверяет, есть ли уже результат для данного task_id."""
        with self._lock:
            return task_id in self._results

    def run(self) -> None:
        while not self._stop:
            try:
                msg = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                if self._stop:
                    break
                continue

            task_id = msg.get("task_id")
            if task_id is None:
                continue

            with self._lock:
                self._results[task_id] = msg
                event = self._waiters.pop(task_id, None)

            if event is not None:
                event.set()

    def stop(self) -> None:
        self._stop = True


# ---------------------------------------------------------------------------
# HealthMonitor — фоновый поток для мониторинга и перезапуска worker'ов
# ---------------------------------------------------------------------------


class _HealthMonitor(threading.Thread):
    """
    Фоновый поток, который периодически проверяет состояние worker-процессов
    и перезапускает упавшие.

    Также отслеживает задачи, которые были в обработке у упавших worker'ов,
    и сигналит об их провале, чтобы вызывающий мог retry.
    """

    def __init__(self, pool: "ASRWorkerPool"):
        super().__init__(daemon=True, name="asr-health-monitor")
        self._pool = pool
        self._stop_event = threading.Event()
        self._restart_counts: dict[int, int] = {i: 0 for i in range(pool.num_workers)}

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=HEALTH_CHECK_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                self._check_workers()
            except Exception as exc:
                logger.error(f"Health monitor error: {exc}\n{traceback.format_exc()}")

    def _check_workers(self) -> None:
        """Проверяет каждый worker и перезапускает упавшие."""
        pool = self._pool

        for i in range(pool.num_workers):
            proc = pool._processes[i]
            if proc is not None and proc.is_alive():
                continue

            # Worker мёртв
            exitcode = proc.exitcode if proc is not None else None
            pid = proc.pid if proc is not None else None

            if self._restart_counts[i] >= MAX_WORKER_RESTARTS:
                logger.error(
                    f"Worker {i} (pid={pid}) is dead (exitcode={exitcode}) "
                    f"but max restarts ({MAX_WORKER_RESTARTS}) reached — not restarting"
                )
                continue

            self._restart_counts[i] += 1
            restart_num = self._restart_counts[i]

            logger.warning(
                f"Worker {i} (pid={pid}) is dead (exitcode={exitcode}). "
                f"Restarting... (restart #{restart_num}/{MAX_WORKER_RESTARTS})"
            )

            try:
                self._restart_worker(i)
                logger.info(
                    f"Worker {i} restarted successfully "
                    f"(new pid={pool._processes[i].pid}, restart #{restart_num})"
                )
            except Exception as exc:
                logger.error(
                    f"Failed to restart worker {i}: {exc}\n{traceback.format_exc()}"
                )

    def _restart_worker(self, worker_id: int) -> None:
        """Перезапускает один worker-процесс."""
        pool = self._pool

        # Убеждаемся что старый процесс точно мёртв
        old_proc = pool._processes[worker_id]
        if old_proc is not None:
            if old_proc.is_alive():
                logger.warning(f"Worker {worker_id} is still alive during restart, terminating...")
                old_proc.terminate()
                old_proc.join(timeout=10)
                if old_proc.is_alive():
                    old_proc.kill()
                    old_proc.join(timeout=5)
            # Закрываем хэндл (Python 3.7+)
            try:
                old_proc.close()
            except (ValueError, Exception):
                pass

        # Создаём новый ready event
        ready = multiprocessing.Event()
        pool._ready_events[worker_id] = ready

        # Создаём и запускаем новый процесс
        p = multiprocessing.Process(
            target=_worker_main,
            args=(worker_id, pool._task_queue, pool._result_queue, ready),
            name=f"asr-worker-{worker_id}",
            daemon=True,
        )
        pool._processes[worker_id] = p
        p.start()

        # Ждём загрузки модели
        ready.wait(timeout=MODEL_LOAD_TIMEOUT)
        if not ready.is_set():
            logger.error(
                f"Restarted worker {worker_id} (pid={p.pid}) "
                f"did not become ready in {MODEL_LOAD_TIMEOUT}s"
            )
        elif not p.is_alive():
            logger.error(
                f"Restarted worker {worker_id} (pid={p.pid}) "
                f"died during model loading (exitcode={p.exitcode})"
            )

    def get_restart_counts(self) -> dict[int, int]:
        """Возвращает количество перезапусков для каждого worker-слота."""
        return dict(self._restart_counts)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# ASRWorkerPool — главный класс
# ---------------------------------------------------------------------------


class ASRWorkerPool:
    """
    Пул ASR worker-процессов для параллельного инференса.

    Каждый worker — отдельный OS-процесс со своей моделью (без GIL).
    Задачи распределяются через общую multiprocessing.Queue,
    результаты приходят через отдельную result_queue.

    Features:
        - Автоматический перезапуск упавших worker-процессов.
        - Таймаут на ожидание результата (TRANSCRIBE_TIMEOUT).
        - Retry задач при сбое worker'а (MAX_TASK_RETRIES).
        - Мониторинг здоровья worker'ов (HEALTH_CHECK_INTERVAL).

    Attributes:
        num_workers: Количество worker-процессов.
    """

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self._processes: list[Optional[multiprocessing.Process]] = []
        self._ready_events: list[multiprocessing.Event] = []

        # Общая очередь задач — worker'ы конкурируют за задачи
        self._task_queue: multiprocessing.Queue = multiprocessing.Queue()
        # Очередь результатов — все worker'ы пишут сюда
        self._result_queue: multiprocessing.Queue = multiprocessing.Queue()

        # Dispatcher разбирает результаты в главном процессе
        self._dispatcher = _ResultDispatcher(self._result_queue)

        # Health monitor
        self._health_monitor = _HealthMonitor(self)

        # Статистика
        self._stats_lock = threading.Lock()
        self._total_requests = 0
        self._active_requests = 0
        self._failed_requests = 0
        self._retried_requests = 0
        self._timed_out_requests = 0

        # Отслеживание in-flight задач (для retry при смерти worker'а)
        self._inflight_lock = threading.Lock()
        # task_id -> original task message (with audio bytes) for potential retry
        self._inflight_tasks: dict[str, dict] = {}

        logger.info(f"Creating ASR worker pool with {num_workers} process(es)")
        logger.info(
            f"Config: TRANSCRIBE_TIMEOUT={TRANSCRIBE_TIMEOUT}s, "
            f"HEALTH_CHECK_INTERVAL={HEALTH_CHECK_INTERVAL}s, "
            f"MAX_TASK_RETRIES={MAX_TASK_RETRIES}, "
            f"MAX_WORKER_RESTARTS={MAX_WORKER_RESTARTS}"
        )

        for i in range(num_workers):
            ready = multiprocessing.Event()
            self._ready_events.append(ready)
            p = multiprocessing.Process(
                target=_worker_main,
                args=(i, self._task_queue, self._result_queue, ready),
                name=f"asr-worker-{i}",
                daemon=True,
            )
            self._processes.append(p)

    def load_all(self) -> None:
        """Запускает все worker-процессы и ждёт загрузки моделей."""
        # Запускаем dispatcher
        self._dispatcher.start()

        # Запускаем worker-процессы
        for i, p in enumerate(self._processes):
            logger.info(f"Starting worker process {i}...")
            p.start()
            logger.info(f"Worker process {i} started (pid={p.pid})")

        # Ждём загрузки моделей во всех worker'ах
        for i, ready in enumerate(self._ready_events):
            logger.info(f"Waiting for worker {i} to load model...")
            ready.wait(timeout=MODEL_LOAD_TIMEOUT)
            if ready.is_set():
                logger.info(f"Worker {i} ready")
            else:
                logger.error(f"Worker {i} did not become ready in time")

        alive = sum(1 for p in self._processes if p is not None and p.is_alive())
        logger.info(f"All workers started: {alive}/{self.num_workers} alive")

        # Запускаем health monitor
        self._health_monitor.start()
        logger.info("Health monitor started")

    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Any:
        """
        Отправляет задачу в пул worker-процессов и ждёт результата.

        Audio передаётся как bytes через multiprocessing.Queue.
        Результат десериализуется из dict обратно в TranscriptionResponse.

        При таймауте или смерти worker'а задача может быть переотправлена
        (до MAX_TASK_RETRIES раз).

        Raises:
            RuntimeError: если задача не удалась после всех попыток.
            TimeoutError: если таймаут истёк и retry исчерпаны.
        """
        # Проверяем что есть живые worker'ы
        alive_count = sum(1 for p in self._processes if p is not None and p.is_alive())
        if alive_count == 0:
            raise RuntimeError(
                "No alive worker processes in pool. "
                "All workers may have crashed. Check logs for details."
            )

        with self._stats_lock:
            self._total_requests += 1
            self._active_requests += 1
            active = self._active_requests

        try:
            return self._transcribe_with_retry(
                audio=audio,
                task=task,
                language=language,
                word_timestamps=word_timestamps,
                output=output,
                options=options,
                active_display=active,
            )
        finally:
            with self._stats_lock:
                self._active_requests -= 1

    def _transcribe_with_retry(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict],
        active_display: int,
    ) -> Any:
        """Выполняет transcribe с retry при сбоях."""
        last_error: Optional[Exception] = None

        for attempt in range(MAX_TASK_RETRIES + 1):
            task_id = uuid.uuid4().hex[:12]

            if attempt > 0:
                with self._stats_lock:
                    self._retried_requests += 1
                logger.warning(
                    f"Retrying transcription task_id={task_id} "
                    f"(attempt {attempt + 1}/{MAX_TASK_RETRIES + 1})"
                )

                # Даём время health monitor'у перезапустить worker
                time.sleep(min(attempt * 2, 10))

                # Проверяем что есть живые worker'ы перед retry
                alive_count = sum(
                    1 for p in self._processes if p is not None and p.is_alive()
                )
                if alive_count == 0:
                    raise RuntimeError(
                        "No alive worker processes after retry wait. "
                        "Cannot retry transcription."
                    )

            logger.info(
                f"Transcription requested task_id={task_id} "
                f"(attempt={attempt + 1}, active={active_display}, "
                f"total_workers={self.num_workers})"
            )

            try:
                result = self._submit_and_wait(
                    task_id=task_id,
                    audio=audio,
                    task=task,
                    language=language,
                    word_timestamps=word_timestamps,
                    output=output,
                    options=options,
                    retry_count=attempt,
                )
                return result
            except TimeoutError as exc:
                last_error = exc
                with self._stats_lock:
                    self._timed_out_requests += 1
                logger.warning(
                    f"Transcription task_id={task_id} timed out "
                    f"after {TRANSCRIBE_TIMEOUT}s (attempt {attempt + 1})"
                )
                continue
            except _WorkerDiedError as exc:
                last_error = exc
                logger.warning(
                    f"Transcription task_id={task_id} failed — worker died "
                    f"(attempt {attempt + 1}): {exc}"
                )
                continue
            except RuntimeError as exc:
                # Ошибка внутри worker'а (модель вернула ошибку) —
                # НЕ retry, т.к. скорее всего повторение даст тот же результат
                with self._stats_lock:
                    self._failed_requests += 1
                raise

        # Все попытки исчерпаны
        with self._stats_lock:
            self._failed_requests += 1

        if isinstance(last_error, TimeoutError):
            raise TimeoutError(
                f"Transcription timed out after {MAX_TASK_RETRIES + 1} attempts "
                f"(timeout={TRANSCRIBE_TIMEOUT}s each)"
            ) from last_error
        else:
            raise RuntimeError(
                f"Transcription failed after {MAX_TASK_RETRIES + 1} attempts: "
                f"{last_error}"
            ) from last_error

    def _submit_and_wait(
        self,
        task_id: str,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict],
        retry_count: int = 0,
    ) -> Any:
        """
        Отправляет одну задачу в очередь и ждёт результат с таймаутом.

        Raises:
            TimeoutError: если результат не пришёл за TRANSCRIBE_TIMEOUT.
            _WorkerDiedError: если все worker'ы умерли во время ожидания.
            RuntimeError: если worker вернул ошибку.
        """
        # Регистрируем ожидание результата
        event = self._dispatcher.register(task_id)

        # Формируем и отправляем задачу
        msg = _make_task(
            task_id=task_id,
            audio=audio,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            output=output,
            options=options,
            retry_count=retry_count,
        )

        # Сохраняем в in-flight для отслеживания
        with self._inflight_lock:
            self._inflight_tasks[task_id] = msg

        self._task_queue.put(msg)

        # Ждём результата с таймаутом
        timeout = TRANSCRIBE_TIMEOUT if TRANSCRIBE_TIMEOUT > 0 else None

        if timeout is not None:
            # Ждём с периодическими проверками здоровья worker'ов
            deadline = time.monotonic() + timeout
            while not event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Таймаут!
                    self._dispatcher.cancel(task_id)
                    with self._inflight_lock:
                        self._inflight_tasks.pop(task_id, None)
                    raise TimeoutError(
                        f"Transcription task {task_id} timed out "
                        f"after {timeout}s"
                    )

                # Проверяем живы ли worker'ы каждые 2 секунды
                wait_chunk = min(remaining, 2.0)
                event.wait(timeout=wait_chunk)

                if not event.is_set():
                    # Проверяем не умерли ли все worker'ы
                    alive_count = sum(
                        1 for p in self._processes
                        if p is not None and p.is_alive()
                    )
                    if alive_count == 0:
                        self._dispatcher.cancel(task_id)
                        with self._inflight_lock:
                            self._inflight_tasks.pop(task_id, None)
                        raise _WorkerDiedError(
                            f"All worker processes died while waiting for "
                            f"task {task_id}. Health monitor should restart them."
                        )
        else:
            # Без таймаута, но всё равно периодически проверяем worker'ов
            while not event.is_set():
                event.wait(timeout=5.0)
                if not event.is_set():
                    alive_count = sum(
                        1 for p in self._processes
                        if p is not None and p.is_alive()
                    )
                    if alive_count == 0:
                        self._dispatcher.cancel(task_id)
                        with self._inflight_lock:
                            self._inflight_tasks.pop(task_id, None)
                        raise _WorkerDiedError(
                            f"All worker processes died while waiting for "
                            f"task {task_id}"
                        )

        # Убираем из in-flight
        with self._inflight_lock:
            self._inflight_tasks.pop(task_id, None)

        # Забираем результат
        result_msg = self._dispatcher.get_result(task_id)

        if not result_msg:
            raise RuntimeError(f"No result received for task {task_id}")

        if not result_msg.get("success"):
            error_type = result_msg.get("error_type", "RuntimeError")
            error_msg = result_msg.get("error_msg", "Unknown error in worker")
            raise RuntimeError(f"[{error_type}] {error_msg}")

        # Десериализуем результат
        result_data = result_msg["result"]
        if result_data["type"] == "str":
            return result_data["value"]
        else:
            # Восстанавливаем TranscriptionResponse из dict
            from src.models.schemas import TranscriptionResponse
            return TranscriptionResponse.model_validate(result_data["value"])

    # ------------------------------------------------------------------
    # Интерфейс совместимый с ASRModel (для drop-in подмены)
    # ------------------------------------------------------------------

    def update_activity(self) -> None:
        pass  # Worker'ы сами обновляют activity при transcribe()

    def ensure_model_loaded(self) -> None:
        # Модели загружаются при load_all()
        pass

    def is_loaded(self) -> bool:
        return any(p is not None and p.is_alive() for p in self._processes)

    def release_model(self) -> None:
        logger.info(f"Stopping {self.num_workers} worker processes...")

        # Останавливаем health monitor первым — иначе он будет пытаться
        # перезапускать worker'ов, которых мы останавливаем
        logger.info("Stopping health monitor...")
        self._health_monitor.stop()
        if self._health_monitor.is_alive():
            self._health_monitor.join(timeout=HEALTH_CHECK_INTERVAL + 2)
        logger.info("Health monitor stopped")

        # Отправляем сигнал остановки каждому worker'у
        for _ in self._processes:
            try:
                self._task_queue.put(_STOP)
            except Exception:
                pass

        # Ждём завершения процессов
        for i, p in enumerate(self._processes):
            if p is None:
                continue
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker process {i} (pid={p.pid}) did not stop, terminating...")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logger.error(f"Worker process {i} (pid={p.pid}) still alive after terminate, killing...")
                    p.kill()

        # Останавливаем dispatcher
        self._dispatcher.stop()

        logger.info("All worker processes stopped")

    def get_info(self) -> dict:
        with self._stats_lock:
            active = self._active_requests
            total = self._total_requests
            failed = self._failed_requests
            retried = self._retried_requests
            timed_out = self._timed_out_requests

        with self._inflight_lock:
            inflight_count = len(self._inflight_tasks)

        worker_infos = []
        for i, p in enumerate(self._processes):
            worker_infos.append({
                "worker_id": i,
                "pid": p.pid if p is not None else None,
                "alive": p.is_alive() if p is not None else False,
                "exitcode": p.exitcode if p is not None else None,
            })

        restart_counts = self._health_monitor.get_restart_counts()

        return {
            "pool_size": self.num_workers,
            "active_requests": active,
            "total_requests": total,
            "failed_requests": failed,
            "retried_requests": retried,
            "timed_out_requests": timed_out,
            "inflight_tasks": inflight_count,
            "loaded": self.is_loaded(),
            "class": "ASRWorkerPool",
            "backend": "multiprocessing",
            "config": {
                "transcribe_timeout": TRANSCRIBE_TIMEOUT,
                "health_check_interval": HEALTH_CHECK_INTERVAL,
                "max_task_retries": MAX_TASK_RETRIES,
                "max_worker_restarts": MAX_WORKER_RESTARTS,
            },
            "workers": worker_infos,
            "restart_counts": restart_counts,
        }


# ---------------------------------------------------------------------------
# Внутренние исключения
# ---------------------------------------------------------------------------


class _WorkerDiedError(Exception):
    """Все worker-процессы умерли во время ожидания результата."""
    pass
