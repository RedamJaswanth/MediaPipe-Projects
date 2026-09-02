import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not open webcam.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Hide the upper part of the camera where your face normally appears
    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, 0),
        (width, int(height * 0.45)),
        (0, 0, 0),
        -1
    )

    # Text
    cv2.putText(
        frame,
        "Hand Detection",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Show webcam
    cv2.imshow("Hand Landmarks", frame)

    # Press ESC to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()