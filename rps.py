import cv2
import mediapipe as mp
import time
import random

mphands = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils

hands = mphands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(1)

fingers = {
    "thumb": (4, 2),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18)
}

unknown_images = [
    cv2.imread("assets/unknown1.png"),
    cv2.imread("assets/unknown2.png"),
    cv2.imread("assets/unknown3.png"),
    cv2.imread("assets/unknown4.png"),
    cv2.imread("assets/unknown5.png"),
    cv2.imread("assets/unknown6.png")
]

move_images = {
    "rock": cv2.imread("assets/rock.png"),
    "paper": cv2.imread("assets/paper.png"),
    "scissors": cv2.imread("assets/scissors.png")
}

def totalopenfingers(landmarks):
    count = 0
    for tip, pip in fingers.values():
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return count

def identifymove(landmarks):
    totalfing = totalopenfingers(landmarks)
    if totalfing <= 1:
        return "rock"
    elif totalfing >= 4:
        return "paper"
    elif totalfing in (2, 3):
        return "scissors"
    return "unknown"

def modelview(move):
    dictofmoves = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
    return dictofmoves.get(move, "unknown")

lasttime = 0
cooldown = 0.3

player_move = "unknown"
ai_move = "unknown"
prev_state = "unknown"

current_unknown_img = random.choice(unknown_images)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    cam = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    current_state = "unknown"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = hand.landmark
        if time.time() - lasttime > cooldown:
            detected = identifymove(landmarks)
            if detected != "unknown":
                player_move = detected
                ai_move = modelview(player_move)
                current_state = "known"
            lasttime = time.time()
        mpdraw.draw_landmarks(cam, hand, mphands.HAND_CONNECTIONS)
    else:
        player_move = "unknown"
        ai_move = "unknown"

    if ai_move != "unknown":
        current_state = "known"

    if current_state == "unknown" and prev_state != "unknown":
        current_unknown_img = random.choice(unknown_images)

    prev_state = current_state

    cam = cv2.resize(cam, (640, 480))

    if ai_move == "unknown":
        move_img = current_unknown_img
    else:
        move_img = move_images[ai_move]

    move_img = cv2.resize(move_img, (640, 480))

    cv2.putText(cam, f"Player: {player_move.upper()}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(cam, f"AI: {ai_move.upper()}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    final = cv2.hconcat([cam, move_img])

    cv2.imshow("Rock Paper Scissors", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
