# Документация OAITT

Документы на русском. Обзор проекта на английском - [README.en.md](../README.en.md).

| Документ | О чём |
|---|---|
| [python-service.md](python-service.md) | Python-сервис: установка, конфигурация, API, Docker |
| [macos-app.md](macos-app.md) | Приложение для macOS: запуск, настройки, данные на диске |
| [swift-port.md](swift-port.md) | Нативный Swift/MLX порт: устройство, API, ключевые решения, ограничения |
| [benchmarks.md](benchmarks.md) | Замеры: Python против Swift, стоимость чанка, батчинг, WER и качество нарезки |
| [compare-crispasr.md](compare-crispasr.md) | Сравнение с CrispASR (ggml/Metal): Parakeet TDT v3 и GigaAM-v3, скорости и качество текста |
| [compare-audiocpp.md](compare-audiocpp.md) | Сравнение с audio.cpp (ggml): GigaAM и Supertonic, скорость, WER, нагрузка, OpenAI-API. Вывод - выгоды нет |
| [roadmap.md](roadmap.md) | План работ: ближайшие задачи с опорой на замеры и проверенные отвергнутые идеи |
| [MEMORY_DEBUGGING.md](MEMORY_DEBUGGING.md) | Отладка утечек памяти в Python-сервисе |

Сборка Docker-образов - [DOCKER_BUILD.md](../DOCKER_BUILD.md),
запуск GigaAM в Docker - [DOCKER_GIGAAM.md](../DOCKER_GIGAAM.md).
