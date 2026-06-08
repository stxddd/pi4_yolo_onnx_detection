import cv2
import numpy as np
from config import INPUT_SIZE, CONFIDENCE, NMS_THRESHOLD, COCO_CLASSES


def preprocess(frame):
    """Готовит кадр для ONNX"""

    height, width = frame.shape[:2]
    new_shape = (INPUT_SIZE, INPUT_SIZE)

    scale = min(new_shape[0] / height, new_shape[1] / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    pad_width = new_shape[1] - resized_width
    pad_height = new_shape[0] - resized_height
    pad_left = pad_width / 2
    pad_top = pad_height / 2

    if (width, height) != (resized_width, resized_height):
        frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_top - 0.1))
    bottom = int(round(pad_height - pad_top - 0.1))
    left = int(round(pad_left - 0.1))
    right = int(round(pad_width - pad_left - 0.1))

    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    tensor = np.expand_dims(frame, axis=0)
    return tensor, scale, pad_left, pad_top


def infer_onnx(session, frame):
    """Выполняет инференс ONNX и возвращает список детекций"""

    tensor, scale, pad_left, pad_top = preprocess(frame)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    predictions = outputs[0]
    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.transpose(1, 0)

    if predictions.shape[1] < 6:
        return []

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep_mask = scores >= CONFIDENCE
    if not np.any(keep_mask):
        return []

    boxes_xywh = boxes_xywh[keep_mask]
    class_ids = class_ids[keep_mask]
    scores = scores[keep_mask]

    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes_xyxy[:, [0, 2]] -= pad_left
    boxes_xyxy[:, [1, 3]] -= pad_top
    boxes_xyxy /= scale

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, frame.shape[1] - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, frame.shape[0] - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, frame.shape[1] - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, frame.shape[0] - 1)

    boxes_for_nms = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        boxes_for_nms.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    nms_indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), CONFIDENCE, NMS_THRESHOLD)
    if len(nms_indices) == 0:
        return []

    if isinstance(nms_indices, np.ndarray):
        nms_indices = nms_indices.flatten().tolist()
    else:
        nms_indices = [index[0] if isinstance(index, (list, tuple)) else int(index) for index in nms_indices]

    detections = []
    for index in nms_indices:
        box = boxes_xyxy[index]
        x1, y1, x2, y2 = [int(value) for value in box]
        class_id = int(class_ids[index])
        score = float(scores[index])
        name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        detections.append((name, score, x1, y1, x2, y2))

    return detections
