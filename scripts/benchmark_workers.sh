#!/bin/bash
# Бенчмарк с разным количеством воркеров

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Активируем venv
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

echo "========================================="
echo "OAITT Multi-Worker Benchmark"
echo "========================================="
echo ""

# Убедимся что порт свободен
PID=$(lsof -ti :9007 2>/dev/null || echo "")
if [ -n "$PID" ]; then
    echo "Killing process $PID on port 9007..."
    kill -9 $PID 2>/dev/null || true
    sleep 2
fi

# Тестовый аудио файл
TEST_AUDIO="$PROJECT_ROOT/sample-data/Sobolev_Andrey_1_0_00-2_17.ogg"
if [ ! -f "$TEST_AUDIO" ]; then
    echo "Test audio not found: $TEST_AUDIO"
    exit 1
fi

# Функция для запуска бенчмарка с заданным количеством воркеров
run_benchmark() {
    local workers=$1
    local iterations=$2
    
    echo ""
    echo "========================================="
    echo "Testing with $workers worker(s)"
    echo "========================================="
    
    # Убиваем предыдущий сервер
    PID=$(lsof -ti :9007 2>/dev/null || echo "")
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null || true
        sleep 2
    fi
    
    # Запускаем сервер
    echo "Starting server with $workers worker(s)..."
    ASR_ENGINE=gigaam \
    GIGAAM_MODEL=v3_e2e_ctc \
    MODEL_WORKERS=$workers \
    PYTHONPATH="${PROJECT_ROOT}/vendor/gigaam:${PYTHONPATH}" \
    python main.py > /tmp/server_${workers}.log 2>&1 &
disown
    SERVER_PID=$!
    
    # Ждем сервер
    echo "Waiting for server..."
    for i in {1..60}; do
        if curl -s http://localhost:9007/health > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:9007/health > /dev/null 2>&1; then
        echo "Server failed to start!"
        tail -20 /tmp/server_${workers}.log
        return 1
    fi
    
    echo "Server ready! Running $iterations iterations..."
    echo ""
    
    # Запускаем тест
    total_time=0
    for i in $(seq 1 $iterations); do
        start_time=$(date +%s.%N)
        
        curl -s -X POST \
            http://localhost:9007/v1/audio/transcriptions \
            -H "Authorization: Bearer key" \
            -F "file=@$TEST_AUDIO" \
            -F "model=whisper-1" \
            -F "language=ru" \
            > /dev/null 2>&1
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $duration" | bc)
        
        if [ $((i % 10)) -eq 0 ]; then
            echo "  Progress: $i/$iterations"
        fi
    done
    
    avg_time=$(echo "scale=3; $total_time / $iterations" | bc)
    throughput=$(echo "scale=2; $iterations / $total_time" | bc)
    
    echo ""
    echo "Results for $workers worker(s):"
    echo "  Total time: ${total_time}s"
    echo "  Avg per request: ${avg_time}s"
    echo "  Throughput: ${throughput} req/s"
    
    # Останавливаем сервер
    kill -9 $SERVER_PID 2>/dev/null || true
    sleep 2
    
    echo "$workers,$avg_time,$throughput" >> /tmp/benchmark_results.csv
}

# Очищаем результаты
rm -f /tmp/benchmark_results.csv
echo "workers,avg_time,throughput" > /tmp/benchmark_results.csv

# Запускаем тесты
echo "Testing sequential requests (single worker)..."
run_benchmark 1 30

echo ""
echo "Testing with 2 workers..."
run_benchmark 2 30

echo ""
echo "Testing with 4 workers..."
run_benchmark 4 30

echo ""
echo "========================================="
echo "Summary"
echo "========================================="
column -s, -t /tmp/benchmark_results.csv

echo ""
echo "Benchmark complete!"
