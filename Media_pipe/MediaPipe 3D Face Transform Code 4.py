import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Output folder
output_folder = r"C:\Users\user\Documents\VS Code Work\Media_Pipe\outputs"
os.makedirs(output_folder, exist_ok=True)

# Output image
output_path = os.path.join(
    output_folder,
    "change_cloth_color_output.png"
)

# Initialize MediaPipe Pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),
    thickness=2,
    circle_radius=2
)

# Color range to replace (BGR)
lower_color = np.array([0, 0, 0])
upper_color = np.array([50, 50, 50])

# New color = RED
new_color = np.array([0, 0, 255])

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press S to save the output image.")
print("Press Q to quit.")

while cap.isOpened():

    success, image = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror camera
    image = cv2.flip(image, 1)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Process pose
    results = pose.process(image_rgb)

    # Start with original image
    result = image.copy()

    if results.pose_landmarks:

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            result,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            drawing_spec,
            drawing_spec
        )

        # Create mask for dark colors
        mask = cv2.inRange(
            image,
            lower_color,
            upper_color
        )

        # Replace dark pixels with red
        result[mask > 0] = new_color

        # Draw landmarks again so they remain visible
        mp_drawing.draw_landmarks(
            result,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            drawing_spec,
            drawing_spec
        )

    # Show output
    cv2.imshow(
        "Change Cloth Color",
        result
    )

    key = cv2.waitKey(1) & 0xFF

    # Press S to save
    if key == ord('s'):

        cv2.imwrite(
            output_path,
            result
        )

        print(
            f"Output saved to: {output_path}"
        )

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()