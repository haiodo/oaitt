"""
OAITT — Open AI Transformer Transcriber.

Сервис мониторинга использования памяти.
Периодически логирует использование RAM и VRAM для диагностики утечек памяти.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import os
import threading
import time
import tracemalloc
from typing import Optional

import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

from src.config import (
    MEMORY_LOG_INTERVAL,
    MEMORY_LOG_ENABLED,
    MEMORY_LOG_TOP_ALLOCATIONS,
)

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Монитор использования памяти с периодическим логированием.
    
    Отслеживает:
    - RSS (Resident Set Size) процесса
    - VMS (Virtual Memory Size) процесса
    - VRAM (GPU memory) если доступно
    - Топ allocation sites если включен tracemalloc
    
    Attributes:
        interval: Интервал логирования в секундах.
        enabled: Включен ли мониторинг.
        top_allocations: Количество топ allocation sites для вывода.
        _stop_event: Event для остановки мониторинга.
        _thread: Поток мониторинга.
        _process: Объект psutil.Process для текущего процесса.
        _snapshot: Предыдущий снапшот tracemalloc для сравнения.
        
    Example:
        >>> monitor = MemoryMonitor()
        >>> monitor.start()
        >>> # ... run your code ...
        >>> monitor.stop()
    """
    
    def __init__(
        self,
        interval: Optional[int] = None,
        enabled: Optional[bool] = None,
        top_allocations: Optional[int] = None,
    ) -> None:
        """
        Инициализирует монитор памяти.
        
        Args:
            interval: Интервал логирования в секундах (default: MEMORY_LOG_INTERVAL).
            enabled: Включен ли мониторинг (default: MEMORY_LOG_ENABLED).
            top_allocations: Количество топ allocation sites (default: MEMORY_LOG_TOP_ALLOCATIONS).
        """
        self.interval = interval if interval is not None else MEMORY_LOG_INTERVAL
        self.enabled = enabled if enabled is not None else MEMORY_LOG_ENABLED
        self.top_allocations = top_allocations if top_allocations is not None else MEMORY_LOG_TOP_ALLOCATIONS
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid())
        self._snapshot: Optional[tracemalloc.Snapshot] = None
        
    def start(self) -> None:
        """Запускает мониторинг памяти в отдельном потоке."""
        if not self.enabled:
            logger.info("Memory monitoring is disabled")
            return
            
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Memory monitor is already running")
            return
            
        # Start tracemalloc if top allocations are requested
        if self.top_allocations > 0:
            try:
                tracemalloc.start()
                self._snapshot = tracemalloc.take_snapshot()
                logger.info("Tracemalloc started for allocation tracking")
            except Exception as e:
                logger.warning(f"Failed to start tracemalloc: {e}")
                
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Memory monitor started with interval={self.interval}s")
        
    def stop(self) -> None:
        """Останавливает мониторинг памяти."""
        if self._thread is None or not self._thread.is_alive():
            return
            
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 1)
        
        # Stop tracemalloc
        if self.top_allocations > 0:
            try:
                tracemalloc.stop()
            except Exception:
                pass
                
        logger.info("Memory monitor stopped")
        
    def _monitor_loop(self) -> None:
        """Основной цикл мониторинга памяти."""
        while not self._stop_event.is_set():
            try:
                self._log_memory_stats()
            except Exception as e:
                logger.error(f"Error logging memory stats: {e}")
                
            # Wait for interval or until stopped
            self._stop_event.wait(self.interval)
            
    def _log_memory_stats(self) -> None:
        """Логирует текущую статистику использования памяти."""
        # Get process memory info
        mem_info = self._process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)
        
        # Get GPU memory info if available
        gpu_info = self._get_gpu_memory_info()
        
        # Format log message
        log_msg = f"Memory: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB"
        if gpu_info:
            log_msg += f", GPU={gpu_info}"
            
        logger.info(log_msg)
        
        # Log tracemalloc top allocations if enabled
        if self.top_allocations > 0 and self._snapshot is not None:
            try:
                self._log_top_allocations()
            except Exception as e:
                logger.debug(f"Failed to log top allocations: {e}")
                
    def _get_gpu_memory_info(self) -> Optional[str]:
        """
        Получает информацию о памяти GPU.
        
        Returns:
            Строка с информацией о памяти GPU или None.
        """
        if not TORCH_AVAILABLE:
            return None
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                return f"allocated={allocated:.1f}MB, reserved={reserved:.1f}MB"
            elif torch.backends.mps.is_available():
                # MPS doesn't have a simple memory API like CUDA
                return "MPS=available"
        except Exception as e:
            logger.debug(f"Failed to get GPU memory info: {e}")
            
        return None
        
    def _log_top_allocations(self) -> None:
        """Логирует топ allocation sites с момента последнего снапшота."""
        try:
            current = tracemalloc.take_snapshot()
            if self._snapshot is not None:
                diff = current.compare_to(self._snapshot, 'lineno')
                
                top = diff[:self.top_allocations]
                for stat in top:
                    logger.info(f"  {stat}")
                
            # Update snapshot for next comparison
            self._snapshot = current
        except Exception as e:
            logger.debug(f"Failed to take tracemalloc snapshot: {e}")
            
    def take_snapshot(self) -> Optional[tracemalloc.Snapshot]:
        """
        Создает снапшот текущего состояния памяти.
        
        Returns:
            Снапшот tracemalloc или None если tracemalloc не запущен.
        """
        if not tracemalloc.is_tracing():
            logger.warning("Tracemalloc is not running, cannot take snapshot")
            return None
            
        try:
            return tracemalloc.take_snapshot()
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            return None
            
    def save_snapshot(self, filepath: str) -> bool:
        """
        Сохраняет текущий снапшот памяти в файл.
        
        Args:
            filepath: Путь к файлу для сохранения.
            
        Returns:
            True если снапшот успешно сохранен.
        """
        snapshot = self.take_snapshot()
        if snapshot is None:
            return False
            
        try:
            snapshot.dump(filepath)
            logger.info(f"Memory snapshot saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save memory snapshot to {filepath}: {e}")
            return False


# Global memory monitor instance
memory_monitor = MemoryMonitor()


def start_memory_monitor() -> None:
    """Запускает глобальный монитор памяти."""
    memory_monitor.start()
    

def stop_memory_monitor() -> None:
    """Останавливает глобальный монитор памяти."""
    memory_monitor.stop()
    

def save_memory_snapshot(filepath: str) -> bool:
    """
    Сохраняет текущий снапшот памяти в файл.
    
    Args:
        filepath: Путь к файлу для сохранения.
        
    Returns:
        True если снапшот успешно сохранен.
    """
    return memory_monitor.save_snapshot(filepath)
