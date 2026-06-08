import time
import cv2

import onnxruntime as ort
from config import CAMERA_INDEX, WIDTH, HEIGHT, INFER_EVERY_N, MODEL_PATH
from camera import open_camera
from preprocessing import infer_onnx
from drawing import draw_detections


def main():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])

    capture = open_camera(CAMERA_INDEX)
    if capture is None:
        raise RuntimeError("Не удалось открыть камеру")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    frame_count = 0
    latest_detections = []
    last_infer_time = time.time()

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                continue

            frame_count += 1
            if frame_count % INFER_EVERY_N == 0:
                now = time.time()
                elapsed = now - last_infer_time
                fps = INFER_EVERY_N / elapsed if elapsed > 0 else 0.0
                last_infer_time = now

                latest_detections = infer_onnx(session, frame)
                timestamp = time.strftime("%H:%M:%S")

                if latest_detections:
                    objects = [f"{name} {score:.0%}" for name, score, *_ in latest_detections]
                    print(f"{timestamp} | FPS: {fps:.1f} | {', '.join(objects)}")
                else:
                    print(f"{timestamp} | FPS: {fps:.1f} | -")

            preview = draw_detections(frame, latest_detections)
            cv2.imshow("YOLO8n (onnx)", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    except KeyboardInterrupt:
        ...
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()