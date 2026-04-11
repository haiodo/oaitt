# Сборка Docker образа OAITT с GigaAM

## Подготовка (один раз)

```bash
# 1. Убедитесь что модели скачаны
ls data/gigaam/
# Должно быть: v3_e2e_rnnt.ckpt, v3_e2e_ctc.ckpt и токенизаторы

# Если нет - запустите:
./prepare.sh
```

## Сборка с помощью build.sh (рекомендуется)

### Простая сборка (текущая платформа)

```bash
./build.sh myuser/oaitt-gigaam 1.0.0
```

### Сборка для конкретной платформы

```bash
# Только AMD64 (x86_64)
./build.sh --amd64 myuser/oaitt-gigaam 1.0.0

# Только ARM64 (Apple Silicon, AWS Graviton)
./build.sh --arm64 myuser/oaitt-gigaam 1.0.0
```

### Мульти-платформенная сборка и публикация

```bash
# Собрать для обеих платформ и запушить
./build.sh --amd64 --arm64 --push myuser/oaitt-gigaam 1.0.0

# Или с тегом latest
./build.sh --amd64 --arm64 --push myuser/oaitt-gigaam latest
```

### Использование кастомного Dockerfile

```bash
./build.sh --file Dockerfile.cpu --amd64 --push myuser/oaitt-gigaam 1.0.0
```

## Ручная сборка (без build.sh)

### Стандартная сборка (10-15 минут)

```bash
docker build -f Dockerfile.cpu -t oaitt-gigaam:cpu .
```

### Сборка с прогрессом

```bash
# Показывать прогресс сборки
DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile.cpu -t oaitt-gigaam:cpu .
```

### Если нужно пересобрать с нуля

```bash
# Без использования кэша
docker build --no-cache -f Dockerfile.cpu -t oaitt-gigaam:cpu .
```

### Мульти-платформенная сборка вручную

```bash
# Создать buildx builder (один раз)
docker buildx create --name oaitt-builder --use

# Собрать и запушить для обеих платформ
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag myuser/oaitt-gigaam:1.0.0 \
    --file Dockerfile.cpu \
    --push .
```

## Проверка образа

```bash
# Размер образа
docker images oaitt-gigaam:cpu

# Запуск для теста
docker run -it --rm -p 9007:9007 oaitt-gigaam:cpu

# В другом терминале:
curl http://localhost:9007/health
```

## Запуск

```bash
# Через docker-compose
docker-compose -f docker-compose.cpu.yml up -d

# Или напрямую
docker run -d \
    --name oaitt-gigaam-cpu \
    -p 9007:9007 \
    -e DEVICE=cpu \
    -e ASR_ENGINE=gigaam \
    -v $(pwd)/data:/app/data \
    oaitt-gigaam:cpu
```

## Проблемы и решения

### Сборка прерывается по timeout

На медленном интернете или ARM64 (Apple Silicon) сборка может занять 20-30 минут.

**Решение:** Запустить сборку в фоне:
```bash
# Запустить и оставить работать
docker build -f Dockerfile.cpu -t oaitt-gigaam:cpu . > build.log 2>&1 &

# Проверять прогресс
tail -f build.log
```

### Медленная загрузка на ARM64

На Apple Silicon pip устанавливает пакеты медленнее.

**Решение:** Использовать `pip install` с флагом `--prefer-binary` (уже включено).

### Не хватает памяти при сборке

```bash
# Ограничить количество параллельных заданий
DOCKER_BUILDKIT=1 docker build --build-arg BUILDKIT_INLINE_CACHE=1 \
    -f Dockerfile.cpu -t oaitt-gigaam:cpu .
```

### Ошибка "multiple platforms feature is currently not supported for docker driver"

**Решение:** Использовать buildx с драйвером docker-container:
```bash
docker buildx create --name multiplatform --driver docker-container --use
docker buildx build --platform linux/amd64,linux/arm64 --push -t myimage:latest .
```

## Публикация в Docker Hub

```bash
# 1. Логин
docker login

# 2. Сборка и публикация
./build.sh --amd64 --arm64 --push myusername/oaitt-gigaam 1.0.0

# 3. Проверка на Docker Hub
# Открыть: https://hub.docker.com/r/myusername/oaitt-gigaam/tags
```

## Альтернатива: Запуск без Docker

Если сборка не работает, можно запустить напрямую:

```bash
# Установить зависимости
pip install -r requirements.txt
pip install -e vendor/gigaam

# Запустить
export ASR_ENGINE=gigaam
export DEVICE=cpu
python main.py
```
