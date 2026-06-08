import platform
import cv2

def open_camera(camera_index):
    """Открывает захват видеопотока по индексу камеры"""

    system_name = platform.system().lower()

    if system_name == "windows":
        candidates = [
            (cv2.CAP_DSHOW),
            (cv2.CAP_MSMF),
            (cv2.CAP_ANY),
        ]
    else:
        candidates = [
            (cv2.CAP_V4L2),
            (cv2.CAP_ANY),
        ]

    for backend in candidates:
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            return capture
        capture.release()

    return None
