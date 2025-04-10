import cv2
import mediapipe as mp

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
frame_w = int(cap.get(3))
frame_h = int(cap.get(4))

# Operation buttons
ops = {
    '+': (100, 100),
    '-': (frame_w - 200, 100),
    '*': (100, frame_h - 100),
    '/': (frame_w - 200, frame_h - 100)
}

current_op = None
result = 0
mode = "select"  # 'select' or 'operate'

# Helpers
def count_fingers(hand_landmarks, handedness):
    tip_ids = [4, 8, 12, 16, 20]
    count = 0
    for tip_id in tip_ids:
        if tip_id == 4:
            if handedness == 'Right':
                if hand_landmarks.landmark[tip_id].x < hand_landmarks.landmark[tip_id - 1].x:
                    count += 1
            else:
                if hand_landmarks.landmark[tip_id].x > hand_landmarks.landmark[tip_id - 1].x:
                    count += 1
        else:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                count += 1
    return count

def is_fist(hand_landmarks):
    tip_ids = [8, 12, 16, 20]
    folded = 0
    for tip in tip_ids:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y:
            folded += 1
    return folded >= 3

def detect_hover(x, y):
    for op, (ox, oy) in ops.items():
        if abs(x - ox) < 60 and abs(y - oy) < 60:
            return op
    return None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_fingers = 0
    right_fingers = 0
    fist_count = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Hand center for hover
            cx = int(handLms.landmark[0].x * frame_w)
            cy = int(handLms.landmark[0].y * frame_h)

            if mode == "select":
                hovered = detect_hover(cx, cy)
                if hovered:
                    current_op = hovered
                    mode = "operate"
                    print(f"Selected operation: {current_op}")
                    break  # Lock selection

            # Count fingers
            count = count_fingers(handLms, label)
            if label == 'Left':
                left_fingers = count
            else:
                right_fingers = count

            # Fist check for exit
            if is_fist(handLms):
                fist_count += 1

    # Exit back to selection if both fists
    if mode == "operate" and fist_count >= 2:
        mode = "select"
        current_op = None
        print("Returning to selection screen...")
        cv2.waitKey(500)

    # Perform operation
    if mode == "operate" and current_op:
        try:
            if current_op == '+':
                result = left_fingers + right_fingers
            elif current_op == '-':
                result = left_fingers - right_fingers
            elif current_op == '*':
                result = left_fingers * right_fingers
            elif current_op == '/':
                result = left_fingers / right_fingers if right_fingers != 0 else 0
        except:
            result = 0

    # Draw UI
    if mode == "select":
        cv2.putText(frame, "Select an operation:", (frame_w // 2 - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for op, (x, y) in ops.items():
            cv2.rectangle(frame, (x - 40, y - 40), (x + 40, y + 40), (100, 200, 100), -1)
            cv2.putText(frame, op, (x - 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    else:
        cv2.putText(frame, f"{left_fingers} {current_op} {right_fingers} = {result}",
                    (50, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.putText(frame, "Fists = Back", (frame_w - 250, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    cv2.imshow("🤜🤛 Finger Math", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
