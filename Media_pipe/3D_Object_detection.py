import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# MediaPipe Objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils


# Load image
image_path = r"C:\Users\user\Pictures\Media Pipe\Mug2.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image could not be loaded.")
    print("Check the image path.")
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Create Objectron
objectron = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.1,
    model_name="Cup"
)


# Run detection
results = objectron.process(image)


# Create output image
annotated_image = image.copy()


# Check detection
if results.detected_objects is None:

    print("No cup detected.")
    print("Try using a clear cup image with good lighting.")

else:

    print(f"Detected {len(results.detected_objects)} cup(s).")

    for detected_object in results.detected_objects:

        # Draw 2D bounding box
        mp_drawing.draw_landmarks(
            annotated_image,
            detected_object.landmarks_2d,
            mp_objectron.BOX_CONNECTIONS
        )

        # Draw 3D axis
        mp_drawing.draw_axis(
            annotated_image,
            detected_object.rotation,
            detected_object.translation
        )


# Display result
plt.figure(figsize=(10, 10))

plt.imshow(annotated_image)

plt.axis("off")

# Save the output image
output_path = r"C:\Users\user\Documents\VS Code Work\Media_Pipe\3D_Object_detection_result.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"Output saved to: {output_path}")

plt.show()


# Close Objectron
objectron.close()