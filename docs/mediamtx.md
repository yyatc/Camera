# MediaMTX

Минимальный запуск локального RTSP-сервера:

1. Скачать [MediaMTX](https://github.com/bluenviron/mediamtx).
2. Запустить `mediamtx`.
3. Убедиться, что порт `8554` доступен.
4. В `config/config.yaml` оставить `stream.output_rtsp = rtsp://127.0.0.1:8554/tracking`.
5. Проверка в VLC:
   - `rtsp://127.0.0.1:8554/tracking`
