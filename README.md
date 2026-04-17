# PTZ Person Tracking (ONVIF + RTSP)

Сервис детектирует человека, управляет PTZ-камерой для удержания цели в центре и публикует RTSP поток с интерфейсом поверх видео.

## Слои
- `src/domain` — бизнес-логика трекинга и PTZ-политика.
- `src/camera` — адаптеры камеры (ONVIF и RTSP input).
- `src/vision` — гибридная детекция и трекер ID.
- `src/stats` — учет времени наблюдения по объектам.
- `src/stream` — рендер оверлея и RTSP output.

## Запуск на новом устройстве (Windows)
1. Установить:
   - Python 3.11+
   - ffmpeg (команда `ffmpeg` должна быть доступна в PATH)
   - MediaMTX (`winget install --id bluenviron.mediamtx -e`)
2. Клонировать проект.
3. Выполнить подготовку:
   - `powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1`
4. Заполнить `config/secrets.env` (создается из `config/secrets.env.example` автоматически).
5. Запустить сервис:
   - `powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1`
6. Проверить состояние:
   - `powershell -ExecutionPolicy Bypass -File .\scripts\status.ps1`
7. Остановить:
   - `powershell -ExecutionPolicy Bypass -File .\scripts\stop.ps1`

### Быстрый запуск через BAT
- `scripts\setup.bat`
- `scripts\start.bat`
- `scripts\status.bat`
- `scripts\stop.bat`

## Поток и проверка
- RTSP URL: `rtsp://127.0.0.1:8554/tracking`
- VLC: `Media -> Open Network Stream`
- Для меньшей задержки в VLC используйте `:rtsp-tcp`.

## Что отображается на видео
- Белый прямоугольник вокруг текущей цели.
- Справа под рамкой: `ID цели / total`.
- В правом верхнем углу:
  - первая строка `Total: чч:мм:сс`,
  - далее список `ID: N sec` по каждому обнаруженному объекту.

## Логи
- При запуске через `scripts/start.ps1` логи пишутся в `runtime-logs/`.
- Для каждой сессии создаются:
  - `session-<timestamp>.txt`
  - `mediamtx-<timestamp>.out.log`, `mediamtx-<timestamp>.err.log`
  - `app-<timestamp>.out.log`, `app-<timestamp>.err.log`
