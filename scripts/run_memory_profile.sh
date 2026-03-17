#!/bin/bash
# OAITT Memory Leak Detection Script for GigaAM ASR
# Запускает бенчмарк с включенным профайлингом памяти для поиска утечек

set -e

# Директория проекта
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Убедимся что порт свободен
echo "Checking port 9007..."
PID=$(lsof -ti :9007 2>/dev/null || echo "")
if [ -n "$PID" ]; then
    echo "Port 9007 is in use by PID $PID, killing..."
    kill -9 $PID 2>/dev/null || true
    sleep 2
fi

# Активируем venv
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ Activated virtual environment"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Activated virtual environment"
else
    echo "⚠️  Warning: No virtual environment found, using system Python"
fi

# Настройки профайлинга памяти
export MEMORY_LOG_ENABLED=true
export MEMORY_LOG_INTERVAL=5  # Логировать каждые 5 секунд для детального отслеживания
export MEMORY_LOG_TOP_ALLOCATIONS=10  # Показывать топ 10 allocation sites

# Настройки GigaAM
export ASR_ENGINE=gigaam
export GIGAAM_MODEL=v3_e2e_ctc
export PYTHONPATH="${PROJECT_ROOT}/vendor/gigaam:${PYTHONPATH}"

# Включаем адаптивный таймаут
export TIMEOUT_ENABLED=true

echo "========================================="
echo "OAITT Memory Leak Detection - GigaAM ASR"
echo "========================================="
echo ""
echo "Configuration:"
echo "  MEMORY_LOG_ENABLED=$MEMORY_LOG_ENABLED"
echo "  MEMORY_LOG_INTERVAL=$MEMORY_LOG_INTERVAL"
echo "  MEMORY_LOG_TOP_ALLOCATIONS=$MEMORY_LOG_TOP_ALLOCATIONS"
echo "  ASR_ENGINE=$ASR_ENGINE"
echo "  GIGAAM_MODEL=$GIGAAM_MODEL"
echo "  TIMEOUT_ENABLED=$TIMEOUT_ENABLED"
echo ""

# Создаем директорию для логов
LOG_DIR="$PROJECT_ROOT/logs/memory_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Запускаем сервер вручную с профайлингом
SERVER_LOG="$LOG_DIR/server.log"
BENCHMARK_LOG="$LOG_DIR/benchmark.log"

echo "Starting GigaAM server with memory profiling..."
echo "Server logs: $SERVER_LOG"
echo ""

# Запускаем сервер в фоне
python main.py > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Ждем пока сервер запустится
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:9007/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        break
    fi
    sleep 1
done

# Проверяем что сервер запустился
if ! curl -s http://localhost:9007/health > /dev/null 2>&1; then
    echo "✗ Server failed to start!"
    tail -50 "$SERVER_LOG"
    exit 1
fi

echo ""
echo "Running benchmark with 50 iterations for memory leak detection..."
echo "This will take a while. Press Ctrl+C to stop early."
echo ""

# Создаем тестовый аудио файл (используем полный файл для тяжелой нагрузки)
TEST_AUDIO="$PROJECT_ROOT/sample-data/Sobolev_Andrey_1_0_00-2_17.ogg"
if [ ! -f "$TEST_AUDIO" ]; then
    echo "✗ Test audio file not found: $TEST_AUDIO"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "Using test audio: $TEST_AUDIO"
echo ""

# Запускаем множество итераций для выявления утечек
ITERATIONS=50
for i in $(seq 1 $ITERATIONS); do
    echo "=== Iteration $i/$ITERATIONS ==="
    
    # Делаем запрос на транскрипцию
    START_TIME=$(date +%s)
    
    curl -s -X POST \
        http://localhost:9007/v1/audio/transcriptions \
        -H "Authorization: Bearer key" \
        -F "file=@$TEST_AUDIO" \
        -F "model=whisper-1" \
        -F "language=ru" \
        -F "response_format=verbose_json" \
        > /dev/null 2>&1
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "Request took ${DURATION}s"
    
    # Делаем небольшую паузу между запросами
    sleep 2
done

echo ""
echo "Benchmark completed! Stopping server..."

# Останавливаем сервер
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "========================================="
echo "Memory Analysis"
echo "========================================="
echo ""

# Анализируем логи на предмет роста памяти
echo "Memory usage trend (RSS in MB):"
grep -E "Memory: RSS=" "$SERVER_LOG" | sed 's/.*RSS=\([0-9.]*\)MB.*/\1/' | nl

echo ""
echo "Top allocation sites over time:"
grep -A 10 "Top allocations" "$SERVER_LOG" | head -150

echo ""
echo "Full server log saved to: $SERVER_LOG"
echo ""
echo "To analyze memory snapshots in detail:"
echo "  1. Check the log file: less $SERVER_LOG"
echo "  2. Look for steadily increasing RSS values"
echo "  3. Check top allocation sites for patterns"
echo ""

# Проверяем, растет ли память
echo "Checking for memory growth..."
MEMORY_VALUES=$(grep -E "Memory: RSS=" "$SERVER_LOG" | sed 's/.*RSS=\([0-9.]*\)MB.*/\1/')
if [ -n "$MEMORY_VALUES" ]; then
    FIRST_MEM=$(echo "$MEMORY_VALUES" | head -1)
    LAST_MEM=$(echo "$MEMORY_VALUES" | tail -1)
    echo "First RSS: ${FIRST_MEM}MB"
    echo "Last RSS: ${LAST_MEM}MB"
    
    # Проверяем рост памяти
    GROWTH=$(echo "$LAST_MEM - $FIRST_MEM" | bc -l 2>/dev/null || echo "0")
    if (( $(echo "$GROWTH > 100" | bc -l 2>/dev/null || echo "0") )); then
        echo "⚠️  WARNING: Memory grew by ${GROWTH}MB - potential leak detected!"
    else
        echo "✓ Memory growth: ${GROWTH}MB (within normal range)"
    fi
fi

echo ""
echo "Analysis complete!"

echo ""
echo "========================================="
echo "Benchmark completed!"
echo "========================================="
echo ""
echo "Analyzing memory usage from logs..."
echo ""

# Анализируем логи на предмет роста памяти
echo "Memory usage trend (RSS in MB):"
grep -E "Memory: RSS=" "$BENCHMARK_LOG" | sed 's/.*RSS=\([0-9.]*\)MB.*/\1/' | nl

echo ""
echo "Top allocation sites over time:"
grep -A 10 "Top allocations" "$BENCHMARK_LOG" | head -100

echo ""
echo "Full log saved to: $BENCHMARK_LOG"
echo ""
echo "To analyze memory snapshots in detail:"
echo "  1. Check the log file: less $BENCHMARK_LOG"
echo "  2. Look for steadily increasing RSS values"
echo "  3. Check top allocation sites for patterns"
echo ""

# Проверяем, растет ли память
echo "Checking for memory growth..."
MEMORY_VALUES=$(grep -E "Memory: RSS=" "$BENCHMARK_LOG" | sed 's/.*RSS=\([0-9.]*\)MB.*/\1/')
if [ -n "$MEMORY_VALUES" ]; then
    FIRST_MEM=$(echo "$MEMORY_VALUES" | head -1)
    LAST_MEM=$(echo "$MEMORY_VALUES" | tail -1)
    echo "First RSS: ${FIRST_MEM}MB"
    echo "Last RSS: ${LAST_MEM}MB"
    
    # Проверяем рост памяти
    GROWTH=$(echo "$LAST_MEM - $FIRST_MEM" | bc -l 2>/dev/null || echo "0")
    if (( $(echo "$GROWTH > 100" | bc -l 2>/dev/null || echo "0") )); then
        echo "⚠️  WARNING: Memory grew by ${GROWTH}MB - potential leak detected!"
    else
        echo "✓ Memory growth: ${GROWTH}MB (within normal range)"
    fi
fi

echo ""
echo "Analysis complete!"
