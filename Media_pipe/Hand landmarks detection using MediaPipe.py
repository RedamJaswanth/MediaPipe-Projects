import cv2
import mediapipe as mp
import os

# --------------------------------------------------
# INITIALIZE MEDIAPIPE HANDS
# --------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# OPEN WEBCAM
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------

output_folder = r"C:\Users\user\Documents\VS Code Work\Media_Pipe\outputs"

os.makedirs(output_folder, exist_ok=True)


# --------------------------------------------------
# OUTPUT VIDEO
# --------------------------------------------------

output_video = os.path.join(
    output_folder,
    "hand_landmarks_output.mp4"
)


# Webcam resolution
width = 640
height = 480
fps = 20.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit()


# --------------------------------------------------
# PROCESS WEBCAM
# --------------------------------------------------

print("Starting Hand Landmarks Detection...")
print("Press ESC to stop.")


while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break


    # Resize frame
    frame = cv2.resize(frame, (width, height))


    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Process with MediaPipe Hands
    results = hands.process(frame_rgb)


    # --------------------------------------------------
    # DRAW HAND LANDMARKS
    # --------------------------------------------------

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw all hand connections
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Draw individual landmark points
            for landmark in hand_landmarks.landmark:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


    # --------------------------------------------------
    # SAVE OUTPUT FRAME
    # --------------------------------------------------

    out.write(frame)


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    cv2.imshow(
        "Hand Landmarks",
        frame
    )


    # ESC key to stop
    if cv2.waitKey(5) & 0xFF == 27:
        break


# --------------------------------------------------
# RELEASE EVERYTHING
# --------------------------------------------------

cap.release()
out.release()
hands.close()
cv2.destroyAllWindows()


# --------------------------------------------------
# OUTPUT MESSAGE
# --------------------------------------------------

print()
print("======================================")
print("HAND LANDMARK DETECTION COMPLETED")
print("======================================")
print("Output video saved to:")
print(output_video)