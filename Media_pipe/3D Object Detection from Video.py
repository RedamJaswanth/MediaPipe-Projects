# ----> VideoCapture(0) tries to capture from the webcam
# ----> cap = cv2.VideoCapture(0)

# ----> 3D OBJECT DETECTION FROM VIDEO

import cv2
import mediapipe as mp
import os

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils


# --------------------------------------------------
# INPUT VIDEO
# --------------------------------------------------

input_video = r"C:\Users\user\Downloads\video-2.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print(input_video)
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
    "3D_Object_detection_video.mp4"
)


# Get video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0


# Create video writer
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
# MEDIAPIPE OBJECTRON
# --------------------------------------------------

objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.70,
    model_name="Cup"
)


# --------------------------------------------------
# READ VIDEO AND DETECT OBJECT
# --------------------------------------------------

print("Processing video...")
print("Press Q to stop.")


while cap.isOpened():

    success, image = cap.read()

    # Stop when video ends
    if not success:
        break

    # Convert BGR → RGB
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = objectron.process(image_rgb)

    # Convert RGB → BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


    # --------------------------------------------------
    # DRAW 3D DETECTION
    # --------------------------------------------------

    if results.detected_objects:

        for detected_object in results.detected_objects:

            mp_drawing.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS
            )

            mp_drawing.draw_axis(
                image,
                detected_object.rotation,
                detected_object.translation
            )


    # --------------------------------------------------
    # SAVE PROCESSED FRAME
    # --------------------------------------------------

    out.write(image)


    # --------------------------------------------------
    # SHOW VIDEO
    # --------------------------------------------------

    cv2.imshow(
        "MediaPipe Objectron",
        cv2.flip(image, 1)
    )


    # Press Q to stop
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# RELEASE
# --------------------------------------------------

cap.release()
out.release()
objectron.close()
cv2.destroyAllWindows()


# --------------------------------------------------
# CHECK OUTPUT
# --------------------------------------------------

print()
print("======================================")
print("VIDEO PROCESSING COMPLETED")
print("======================================")
print("Output saved to:")
print(output_video)