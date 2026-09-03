import cv2, numpy as np, tensorflow as tf
from collections import deque
from itungorang import PoseSwitcher

model = tf.keras.models.load_model("model/boxing_lstm.keras")
classes = np.load("model/classes.npy", allow_pickle=True)
SEQ_LEN, CONF_THRESH = 24, 0.65

@tf.function
def fast_predict(x):
    return model(x, training=False)

_ = fast_predict(tf.zeros((1, SEQ_LEN, 51)))

switcher = PoseSwitcher(check_interval=15)
buffer = deque(maxlen=SEQ_LEN)

PROCESS_WIDTH = 480       
PREDICT_EVERY_N = 3       
frame_idx = 0
label, conf, mode = "...", 0.0, "SINGLE"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    h, w = frame.shape[:2]
    scale = PROCESS_WIDTH / w
    small = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))
    mode, people = switcher.get_people(small)
    if people:
        buffer.append(people[0])
        if mode == "MULTI" and len(people) > 1:
            cv2.putText(frame, "2 people tracked", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if len(buffer) == SEQ_LEN and frame_idx % PREDICT_EVERY_N == 0:
            seq = tf.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
            pred = fast_predict(seq).numpy()[0]
            idx = np.argmax(pred)
            label, conf = classes[idx], pred[idx]
    cv2.putText(frame, f"[{mode}] {label} ({conf:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Boxing Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()