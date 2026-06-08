import cv2


def draw_detections(frame, detections):
    """Рисует обнаруженные объекты на кадре и возвращает аннотированное изображение"""

    annotated = frame.copy()

    for name, score, x1, y1, x2, y2 in detections:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{name} {score:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        text_top = max(0, y1 - text_height - baseline - 6)
        text_bottom = text_top + text_height + baseline + 6
        text_right = min(frame.shape[1] - 1, x1 + text_width + 8)

        cv2.rectangle(
            annotated,
            (x1, text_top),
            (text_right, text_bottom),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, text_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated
