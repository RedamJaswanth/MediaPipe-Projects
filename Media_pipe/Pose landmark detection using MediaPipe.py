import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Step 1: Create Pose detector
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

# Step 2: Load the input image
image_path = r"C:\Users\user\Pictures\Allu New.jpg"

image = cv2.imread("C:\\Users\\user\\Pictures\\Allu New.jpg")

# Check if image was loaded
if image is None:
    print("Error: Image not found.")
    print("Check the image path:")
    print(image_path)
    exit()

# Step 3: Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image
results = detector.process(image_rgb)

# Step 4: Draw pose landmarks
annotated_image = image.copy()

if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 0, 255),
            thickness=2
        )
    )

    print("Pose landmarks detected successfully.")

else:
    print("No person detected in the image.")

# Step 5: Display result
plt.figure(figsize=(10, 10))

plt.imshow(
    cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
)

plt.axis("off")
plt.show()

# Close detector
detector.close()