"""Annotate a video file with frame-by-frame punch predictions.

The script reuses the same pose and sequence pipeline as the webcam
version, writes the prediction overlay to a new video, and counts confident
non-``no_punch`` predictions for a simple summary.
"""

import cv2, numpy as np, tensorflow as tf
from collections import deque, Counter
from itungorang import PoseSwitcher

model = tf.keras.models.load_model("model/boxing_lstm.keras")
classes = np.load("model/classes.npy", allow_pickle=True)
SEQ_LEN, CONF_THRESH = 24, 0.65

def analyze_video(input_path, output_path="annotated_output.mp4"):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    switcher = PoseSwitcher()
    buffer = deque(maxlen=SEQ_LEN)
    punch_log = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        mode, people = switcher.get_people(frame)
        label = "..."
        if people:
            buffer.append(people[0])
            if len(buffer) == SEQ_LEN:
                seq = np.expand_dims(np.array(buffer), axis=0)
                pred = model.predict(seq, verbose=0)[0]
                idx = np.argmax(pred)
                label, conf = classes[idx], pred[idx]
                if label != "no_punch" and conf > CONF_THRESH:
                    punch_log.append(label)

        cv2.putText(frame, f"[{mode}] {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    counts = Counter(punch_log)
    print("=== Punch Summary ===")
    for k, v in counts.items():
        print(f"{k}: {v}")
    return output_path, counts

if __name__ == "__main__":
    analyze_video("uploaded_match.mp4")