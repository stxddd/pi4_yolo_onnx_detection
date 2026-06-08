# Простое распознование объектов YOLO (ONNX)

Программа адаптированная под Rasberry PI 4/5. Инференс модели YOLO8n.

## Установка

- yolov8n.onnx уже есть в корне проекта.

- добавь opencv-python==4.9.0.80 в requiremets.txt для запуска на PC WIN 10\11

```bash
sudo apt update
sudo apt install -y python3-opencv python3-venv python3-pip

python3 -m venv venv --system-site-packages
source venv/bin/activate

pip install -r requirements.txt

python3 app/main.py
```

## Настройки

config.py

| Параметр | По умолчанию | Описание |
|---|---|---|
| `CAMERA_INDEX` | `0` | Индекс камеры (`ls /dev/video*`) |
| `CONFIDENCE` | `0.5` | Порог уверенности (0.0 – 1.0) |
| `NMS_THRESHOLD` | `0.45` | Наложения рамок |
| `WIDTH, HEIGHT` | `320, 240` | Разрешение кадра |
| `INPUT_SIZE` | `320` | Длина стороны вх. изображения |
| `INFER_EVERY_N` | `3` | Инференс каждый N-й кадр |