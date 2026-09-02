import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Output folder
output_folder = r"C:\Users\user\Documents\VS Code Work\Media_Pipe\outputs"
os.makedirs(output_folder, exist_ok=True)

# Output image
output_path = os.path.join(
    output_folder,
    "3d_face_replacement_output.png"
)

# Initialize MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    color=(0, 255, 0)
)


def get_face_bbox(landmarks, image_shape):

    x_coordinates = [landmark[0] for landmark in landmarks]
    y_coordinates = [landmark[1] for landmark in landmarks]

    xmin = int(min(x_coordinates) * image_shape[1])
    ymin = int(min(y_coordinates) * image_shape[0])
    xmax = int(max(x_coordinates) * image_shape[1])
    ymax = int(max(y_coordinates) * image_shape[0])

    # Keep coordinates inside image
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_shape[1], xmax)
    ymax = min(image_shape[0], ymax)

    return xmin, ymin, xmax, ymax


def transform_3d_face(image, landmarks, replacement_face):

    transformed_image = image.copy()

    xmin, ymin, xmax, ymax = get_face_bbox(
        landmarks,
        image.shape[:2]
    )

    # Check valid face region
    if xmax <= xmin or ymax <= ymin:
        return transformed_image

    # Resize replacement face
    resized_replacement_face = cv2.resize(
        replacement_face,
        (xmax - xmin, ymax - ymin)
    )

    # Create mask
    mask = cv2.cvtColor(
        resized_replacement_face,
        cv2.COLOR_BGR2GRAY
    ) / 255.0

    # Region of interest
    roi = transformed_image[
        ymin:ymax,
        xmin:xmax
    ].astype(np.float32)

    replacement = resized_replacement_face.astype(
        np.float32
    )

    # Blend replacement face
    mask = mask[:, :, np.newaxis]

    roi = (
        roi * (1 - mask)
        + replacement * mask
    )

    transformed_image[
        ymin:ymax,
        xmin:xmax
    ] = roi.astype(np.uint8)

    return transformed_image


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture first frame as replacement face
ret, replacement_face = cap.read()

if not ret:
    print("Failed to capture replacement face.")
    cap.release()
    exit()

# Flip replacement face
replacement_face = cv2.flip(
    replacement_face,
    1
)

print("Webcam started.")
print("Press S to save the output image.")
print("Press Q to quit.")

while cap.isOpened():

    success, image = cap.read()

    if not success:
        break

    # Mirror camera
    image = cv2.flip(image, 1)

    # Default output
    transformed_image = image.copy()

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Process face
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # Convert landmarks
            landmarks = [
                (lm.x, lm.y, lm.z)
                for lm in face_landmarks.landmark
            ]

            # Apply face transformation
            transformed_image = transform_3d_face(
                image,
                landmarks,
                replacement_face
            )

            # Draw face mesh
            mp_drawing.draw_landmarks(
                transformed_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Create side-by-side output
    composite_image = np.hstack(
        (replacement_face, transformed_image)
    )

    # Show output
    cv2.imshow(
        "MediaPipe 3D Face Transform",
        composite_image
    )

    key = cv2.waitKey(1) & 0xFF

    # Save output
    if key == ord('s'):

        cv2.imwrite(
            output_path,
            composite_image
        )

        print(
            f"Output saved to: {output_path}"
        )

    # Quit
    if key == ord('q'):
        break


cap.release()
face_mesh.close()
cv2.destroyAllWindows()