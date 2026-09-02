# Face and Hand Landmarks Detection using MediaPipe

import cv2
import time
import mediapipe as mp
import os

# --------------------------------------------------
# MEDIAPIPE HOLISTIC MODEL
# --------------------------------------------------

mp_holistic = mp.solutions.holistic

holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils


# --------------------------------------------------
# WEBCAM
# --------------------------------------------------

capture = cv2.VideoCapture(0)

if not capture.isOpened():
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
    "face_hand_landmarks_output.mp4"
)


# Webcam resolution
width = 800
height = 600

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
    capture.release()
    exit()


# --------------------------------------------------
# FPS
# --------------------------------------------------

previousTime = 0


# --------------------------------------------------
# PROCESS WEBCAM
# --------------------------------------------------

print("Starting webcam...")
print("Press ESC to stop.")


while capture.isOpened():

    ret, frame = capture.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break


    # Resize frame
    frame = cv2.resize(frame, (800, 600))


    # BGR → RGB
    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Improve performance
    image.flags.writeable = False

    results = holistic_model.process(image)

    image.flags.writeable = True


    # RGB → BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------
    # FACE LANDMARKS
    # --------------------------------------------------

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS
    )


    # --------------------------------------------------
    # RIGHT HAND LANDMARKS
    # --------------------------------------------------

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )


    # --------------------------------------------------
    # LEFT HAND LANDMARKS
    # --------------------------------------------------

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )


    # --------------------------------------------------
    # CALCULATE FPS
    # --------------------------------------------------

    currentTime = time.time()

    if previousTime != 0:
        fps_display = 1 / (currentTime - previousTime)
    else:
        fps_display = 0

    previousTime = currentTime


    # Display FPS
    cv2.putText(
        image,
        str(int(fps_display)) + " FPS",
        (10, 70),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------
    # SAVE OUTPUT FRAME
    # --------------------------------------------------

    out.write(image)


    # --------------------------------------------------
    # DISPLAY WEBCAM
    # --------------------------------------------------

    cv2.imshow(
        "Facial and Hand Landmarks",
        image
    )


    # Press ESC to stop
    if cv2.waitKey(5) & 0xFF == 27:
        break


# --------------------------------------------------
# RELEASE EVERYTHING
# --------------------------------------------------

capture.release()
out.release()
holistic_model.close()
cv2.destroyAllWindows()


# --------------------------------------------------
# OUTPUT MESSAGE
# --------------------------------------------------

print()
print("======================================")
print("FACE & HAND LANDMARK DETECTION DONE")
print("======================================")
print("Output video saved to:")
print(output_video)