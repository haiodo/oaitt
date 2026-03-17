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
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

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
    }


def _parse_audio(msg: dict) -> np.ndarray:
    """Восстанавливает numpy array из сериализованной задачи."""
    return np.frombuffer(msg["audio_bytes"], dtype=msg["audio_dtype"]).reshape(
        msg["audio_shape"]
    )


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
# ASRWorkerPool — главный класс
# ---------------------------------------------------------------------------


class ASRWorkerPool:
    """
    Пул ASR worker-процессов для параллельного инференса.

    Каждый worker — отдельный OS-процесс со своей моделью (без GIL).
    Задачи распределяются через общую multiprocessing.Queue,
    результаты приходят через отдельную result_queue.

    Attributes:
        num_workers: Количество worker-процессов.
    """

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self._processes: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.Event] = []

        # Общая очередь задач — worker'ы конкурируют за задачи
        self._task_queue: multiprocessing.Queue = multiprocessing.Queue()
        # Очередь результатов — все worker'ы пишут сюда
        self._result_queue: multiprocessing.Queue = multiprocessing.Queue()

        # Dispatcher разбирает результаты в главном процессе
        self._dispatcher = _ResultDispatcher(self._result_queue)

        # Статистика
        self._stats_lock = threading.Lock()
        self._total_requests = 0
        self._active_requests = 0

        logger.info(f"Creating ASR worker pool with {num_workers} process(es)")

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
            ready.wait(timeout=600)  # 10 минут на загрузку
            if ready.is_set():
                logger.info(f"Worker {i} ready")
            else:
                logger.error(f"Worker {i} did not become ready in time")

        alive = sum(1 for p in self._processes if p.is_alive())
        logger.info(f"All workers started: {alive}/{self.num_workers} alive")

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
        """
        with self._stats_lock:
            self._total_requests += 1
            self._active_requests += 1
            active = self._active_requests

        task_id = uuid.uuid4().hex[:12]

        logger.info(
            f"Transcription requested task_id={task_id} "
            f"(active={active}, total_workers={self.num_workers})"
        )

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
        )
        self._task_queue.put(msg)

        # Ждём результата
        event.wait()

        with self._stats_lock:
            self._active_requests -= 1

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
        return any(p.is_alive() for p in self._processes)

    def release_model(self) -> None:
        logger.info(f"Stopping {self.num_workers} worker processes...")

        # Отправляем сигнал остановки каждому worker'у
        for _ in self._processes:
            try:
                self._task_queue.put(_STOP)
            except Exception:
                pass

        # Ждём завершения процессов
        for i, p in enumerate(self._processes):
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

        worker_infos = []
        for i, p in enumerate(self._processes):
            worker_infos.append({
                "worker_id": i,
                "pid": p.pid,
                "alive": p.is_alive(),
                "exitcode": p.exitcode,
            })

        return {
            "pool_size": self.num_workers,
            "active_requests": active,
            "total_requests": total,
            "loaded": self.is_loaded(),
            "class": "ASRWorkerPool",
            "backend": "multiprocessing",
            "workers": worker_infos,
        }
