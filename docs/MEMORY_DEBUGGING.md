# Диагностика утечек памяти в OAITT

Этот документ описывает методы диагностики утечек памяти в OAITT сервисе.

## Быстрый старт: Мониторинг памяти

### 1. Включение периодического логирования памяти

Запустите сервис с включенным мониторингом памяти:

```bash
# Базовый запуск с мониторингом
MEMORY_LOG_ENABLED=true ./run_gigaam_asr.sh

# С настройкой интервала (каждые 30 секунд)
MEMORY_LOG_ENABLED=true MEMORY_LOG_INTERVAL=30 ./run_gigaam_asr.sh

# С логированием топ allocation sites
MEMORY_LOG_ENABLED=true MEMORY_LOG_TOP_ALLOCATIONS=10 ./run_gigaam_asr.sh
```

### 2. Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `MEMORY_LOG_ENABLED` | Включить логирование памяти | `false` |
| `MEMORY_LOG_INTERVAL` | Интервал логирования в секундах | `60` |
| `MEMORY_LOG_TOP_ALLOCATIONS` | Количество топ allocation sites | `5` |

### 3. Пример вывода логов

```
2024-01-15 10:30:00,123 - src.services.memory_monitor - INFO - Memory: RSS=2048.5MB, VMS=4096.2MB, GPU=allocated=1024.0MB, reserved=1536.0MB
2024-01-15 10:30:00,124 - src.services.memory_monitor - INFO -   <frozen importlib._bootstrap>:228: size=4855 KiB, count=39332, average=126 B
2024-01-15 10:30:00,125 - src.services.memory_monitor - INFO -   /path/to/gigaam/model.py:45: size=1024 KiB, count=1, average=1024 KiB
```

## Анализ утечек на запущенном процессе

Если процесс уже запущен и вы подозреваете утечку памяти:

### Метод 1: Использование memory_profiler

```bash
# Установите memory_profiler
pip install memory_profiler

# Присоединитесь к процессу
python -m memory_profiler --pid <PID>

# Или создайте снапшот
mprof run --python python -c "import os; os.kill(<PID>, 0)"
```

### Метод 2: Использование pympler

```python
# В отдельном Python скрипте
from pympler import tracker, muppy, summary
import gc

tr = tracker.SummaryTracker()

# До операции
tr.print_diff()

# После операции  
tr.print_diff()
```

### Метод 3: Снятие дампа памяти с помощью tracemalloc

Создайте скрипт `dump_memory.py`:

```python
#!/usr/bin/env python3
"""
Скрипт для снятия дампа памяти с запущенного Python процесса.
Требует доступа к процессу и установленного gdb.
"""

import sys
import subprocess
import tempfile
import os

def create_gdb_script(pid, output_file):
    """Создает GDB скрипт для снятия дампа памяти."""
    gdb_script = f"""
set pagination off
attach {pid}
python
import tracemalloc
tracemalloc.start()
snapshot = tracemalloc.take_snapshot()
snapshot.dump("{output_file}")
end
detach
quit
"""
    return gdb_script

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <PID>")
        print(f"Example: {sys.argv[0]} 12345")
        sys.exit(1)
    
    pid = sys.argv[1]
    output_file = f"/tmp/memory_snapshot_{pid}.dat"
    
    # Create temporary GDB script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
        f.write(create_gdb_script(pid, output_file))
        gdb_script_file = f.name
    
    try:
        # Run GDB
        result = subprocess.run(
            ['gdb', '-batch', '-x', gdb_script_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"Memory snapshot saved to: {output_file}")
            print(f"\nTo analyze:")
            print(f"  python3 -c \"import tracemalloc; s = tracemalloc.Snapshot.load('{output_file}'); top = s.statistics('lineno')[:10]; [print(t) for t in top]\"")
        else:
            print(f"Error: {result.stderr}")
            sys.exit(1)
    finally:
        os.unlink(gdb_script_file)

if __name__ == "__main__":
    main()
```

Запуск:
```bash
chmod +x dump_memory.py
./dump_memory.py <PID>
```

### Метод 4: Использование py-spy

```bash
# Установите py-spy
pip install py-spy

# Создайте dump heap
py-spy dump --pid <PID> --locals

# Или запишите профиль на 60 секунд
py-spy record -o profile.svg --pid <PID> --duration 60
```

### Метод 5: Мониторинг через /proc (Linux)

```bash
# Найдите PID процесса
pgrep -f "python main.py"

# Мониторинг RSS в реальном времени
watch -n 5 'cat /proc/<PID>/status | grep -E "VmRSS|VmSize"'

# Или с помощью ps
watch -n 5 'ps -o pid,rss,vsz,comm -p <PID>'
```

## Анализ дампа памяти

### Анализ снапшота tracemalloc

```python
import tracemalloc

# Загрузите снапшот
snapshot = tracemalloc.Snapshot.load("/tmp/memory_snapshot_12345.dat")

# Топ allocation sites по размеру
top_stats = snapshot.statistics('lineno')
print("[ Top 10 allocations ]")
for stat in top_stats[:10]:
    print(stat)

# Топ по количеству объектов
top_stats = snapshot.statistics('lineno', cumulative=False)
print("\n[ Top 10 by count ]")
for stat in top_stats[:10]:
    print(stat)

# Сравнение двух снапшотов
snapshot1 = tracemalloc.Snapshot.load("snapshot1.dat")
snapshot2 = tracemalloc.Snapshot.load("snapshot2.dat")
diff = snapshot2.compare_to(snapshot1, 'lineno')
print("\n[ Differences ]")
for stat in diff[:10]:
    print(stat)
```

### Анализ с помощью meliae

```bash
pip install meliae

# В коде приложения
from meliae import scanner
scanner.dump_all_objects("/tmp/memory_dump.json")

# Анализ
python -m meliae.loader /tmp/memory_dump.json
```

## Специфичные для GigaAM рекомендации

### 1. Проверка очистки CUDA кэша

В `src/asr/gigaam.py` уже реализована очистка памяти после транскрипции:

```python
# Clear memory cache to prevent accumulation over multiple transcriptions
clear_memory_cache()
```

### 2. Мониторинг GPU памяти

Для CUDA:
```bash
# В отдельном терминале
watch -n 1 nvidia-smi

# Или с логированием в файл
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 > gpu_memory.log
```

Для MPS (macOS):
```bash
# Используйте Activity Monitor или
vm_stat 5
```

### 3. Ограничение размера батча

Если утечка связана с большими аудио файлами:

```bash
# Уменьшите размер чанка
export GIGAAM_CHUNK_SEC=20
export GIGAAM_MIN_CHUNK_SEC=3

./run_gigaam_asr.sh
```

## Автоматический анализ с помощью скрипта

Создайте скрипт `analyze_memory.sh`:

```bash
#!/bin/bash
# Анализ использования памяти процессом OAITT

PID=$1
if [ -z "$PID" ]; then
    echo "Usage: $0 <PID>"
    exit 1
fi

echo "Memory analysis for PID $PID"
echo "=============================="

# RSS over time
echo -e "\n=== RSS Memory (MB) ==="
for i in {1..10}; do
    RSS=$(ps -o rss= -p $PID | awk '{print $1/1024}')
    echo "$(date '+%H:%M:%S'): ${RSS} MB"
    sleep 5
done

# GPU memory if available
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n=== GPU Memory ==="
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | grep $PID || echo "No GPU memory used"
fi

# Open files
echo -e "\n=== Open Files ==="
ls -la /proc/$PID/fd/ 2>/dev/null | wc -l
echo "Total open file descriptors"

# Threads
echo -e "\n=== Threads ==="
ps -o nlwp -p $PID
```

## Дополнительные инструменты

### memray

```bash
pip install memray

# Профилирование
memray run -o memory.bin main.py

# Анализ
memray flamegraph memory.bin
memray table memory.bin
```

### scalene

```bash
pip install scalene

# Полное профилирование CPU + память
scalene main.py
```

## Рекомендации

1. **Всегда используйте мониторинг памяти** в production: `MEMORY_LOG_ENABLED=true`
2. **Устанавливайте разумные таймауты** для моделей: `MODEL_IDLE_TIMEOUT=3600`
3. **Ограничивайте размер входящих аудио** на уровне API
4. **Используйте chunked transcription** для длинных аудио
5. **Регулярно перезапускайте** сервис при обнаружении утечек (временное решение)

## Связанные файлы

- `src/services/memory_monitor.py` - Сервис мониторинга памяти
- `src/config.py` - Конфигурация переменных окружения
- `src/asr/gigaam.py` - Очистка памяти после транскрипции
- `src/utils/device.py` - Утилиты для работы с устройствами
