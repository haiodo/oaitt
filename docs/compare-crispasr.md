# Сравнение с CrispASR

CrispASR - форк whisper.cpp: один C++ бинарник на ggml, Metal на macOS, веса GGUF,
более сотни бэкендов, включая Parakeet TDT v3 и GigaAM-v3. Без Python, без MLX.
Сравнивали обе общие модели: Parakeet TDT v3 и GigaAM-v3 (rnnt/ctc).

## Стенд и методика

MacBook Pro (Mac16,5), Apple M4 Max, 48 GB. Аудио - `sample-data/Sobolev_Andrey_1_0_00-2_17.ogg`
(137.4s), для CrispASR переконвертировано в wav 16k mono. Замеры 2026-09-26, медиана
тёплых прогонов, первый холодный отброшен. Разброс на этой машине 10-20%
(см. [benchmarks.md](benchmarks.md)).

- **oaitt** - Swift CLI end-to-end через `time -p`, включая загрузку весов.
  Отсутствие вычета загрузки играет против нас, в пользу CrispASR.
- **CrispASR** - строка `transcribed Ns audio in X.Xs` из stderr: таймер стартует
  после загрузки модели и декодирования аудио, это чистая транскрипция. Методика -
  их `docs/benchmarking.md`, метод 3.
- Тексты всех прогонов побайтово совпадают (детерминизм у обоих).

## Результаты

| Модель | oaitt Swift | oaitt Python | CrispASR |
|---|---|---|---|
| Parakeet TDT v3 | **0.92s** (149x, fp16) | 1.9s (72x, fp16) | 2.14s (64x, f16) / 2.70s (51x, Q4_K) |
| GigaAM-v3 RNNT (plain) | 0.61s (225x, fp16) | 0.82s (168x) | 0.56s (246x, f16) |
| GigaAM-v3 e2e-RNNT (пунктуация) | - | - | 0.54s (255x, f16) |
| GigaAM-v3 CTC | **0.38s** (361x, fp16) | 0.59s (231x) | 0.50s (275x, e2e-CTC f16) |

Python-числа - из [benchmarks.md](benchmarks.md) (ин-процесс, те же файл и стенд).
In-процесс Swift-замеры оттуда же: CTC 329x, RNNT 194x; Parakeet 122.7x (2026-09-14).

## Выводы

- **Parakeet TDT v3: наш Swift быстрее минимум в 2.3 раза** (0.92s end-to-end против
  2.14s чистого инференса; чистый против чистого - заметно больше). У них это узкое
  место по декоду: TDT-декод в ggml медленный (их же PERFORMANCE.md называет v3
  decode-bound), плюс стриминговые чанки 30s с 3s перекрытием и LCS-merge - повторное
  кодирование и склейка. У нас один чанк на весь файл, fp16, собственный TDT-декод.
  Q4_K у них медленнее f16 - деквант на GPU дороже, чем экономия памяти.
- **GigaAM-v3 RNNT (plain): паритет.** 0.61s у нас - с загрузкой весов, их 0.56s -
  без. С вычетом загрузки (~0.2s по разнице холодного и тёплого прогона) мы быстрее,
  но в пределах того же разброса 10-20%.
- **GigaAM-v3 CTC: наш быстрее даже в этой неравной постановке** (0.38s с загрузкой
  против 0.50s без). У энкодера ggml скорость на Metal ниже, чем у MLX-порта.
- Оба e2e-варианта CrispASR (SentencePiece 1024 + ITN) дают пунктуацию и капитализацию
  из коробки, что мы пока делаем отдельными средствами.

## Качество текста на тестовом файле

- Их GigaAM e2e-RNNT: чистый текст с пунктуацией и ITN («GigaChat», «Питон»).
- Их e2e-CTC: ITN ломает числительные - «один, два» превращается в «по в равно».
- Их Parakeet v3 (Q4_K) и наш Parakeet v3: текст сопоставимый, осмысленный.
- Их plain RNNT: без пунктуации, как наш, но на этом файле хуже - «индикаторе»,
  «торе поделать» вместо связного текста; у e2e-RNNT таких провалов нет.

## Наблюдения про CrispASR

- README на GitHub повреждён автогенерацией: цифры и URL в таблицах искажены
  («25 EU» вместо 25x, несуществующие репозитории). Реальные цифры - в PERFORMANCE.md,
  реальные адреса весов - в `src/crispasr_model_registry.cpp`.
- `cmake --build` ломается на тестах (расхождение API теста с ggml-сабмодулем);
  цель `crispasr-cli` собирается. Веса: `cstr/parakeet-tdt-0.6b-v3-GGUF` (только
  квантованные, f16 нет) и `cstr/gigaam-v3-GGUF` (f16/q8_0/q4_k, e2e и plain).
- Для них это общий движок на сотню архитектур, для нас - заточенный порт: на
  Parakeet и GigaAM CTC мы быстрее, на GigaAM RNNT равны, при этом у них больше
  моделей и готовые e2e-варианты с пунктуацией.

## Как воспроизвести

```bash
git clone --recursive --depth 1 https://github.com/CrispStrobe/CrispASR.git
cmake -B CrispASR/build -DCMAKE_BUILD_TYPE=Release && cmake --build CrispASR/build --config Release --target crispasr-cli -j 16
# веса - с huggingface.co/cstr/gigaam-v3-GGUF и cstr/parakeet-tdt-0.6b-v3-GGUF
crispasr --backend gigaam -m gigaam-v3-rnnt-f16.gguf -f audio.wav -otxt -of out
```

oaitt: `swift/.build/release/oaitt-swift transcribe --parakeet-dir data/parakeet_tdt_v3
--max-chunk-sec 200 audio.wav` (GigaAM: `--model-type rnnt|ctc --model-cache-dir data/gigaam_mlx`).
