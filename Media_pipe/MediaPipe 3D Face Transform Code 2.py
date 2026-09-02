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
    "3d_face_transform_output.png"
)

# Initialize MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing specification
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    color=(0, 255, 0)
)

# 3D transformation matrix
transformation_matrix = np.array([
    [1.5, 0, 0],
    [0, 1.5, 0],
    [0, 0, 1]
])


def transform_3d_face(image, landmarks):

    # Perform 3D transformation
    transformed_landmarks = np.matmul(
        landmarks,
        transformation_matrix.T
    )

    transformed_image = image.copy()

    for i in range(transformed_landmarks.shape[0]):

        x, y, z = transformed_landmarks[i]

        x = int(x * image.shape[1])
        y = int(y * image.shape[0])

        # Keep coordinates inside the image
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))

        cv2.circle(
            transformed_image,
            (x, y),
            1,
            (255, 0, 0),
            -1
        )

    return transformed_image


# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():

    success, image = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Flip for selfie view
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

            # Convert landmarks to NumPy array
            landmarks = np.zeros(
                (468, 3),
                dtype=np.float32
            )

            for i, landmark in enumerate(
                face_landmarks.landmark
            ):

                landmarks[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

            # Apply 3D transformation
            transformed_image = transform_3d_face(
                image,
                landmarks
            )

            # Draw face mesh
            mp_drawing.draw_landmarks(
                transformed_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Display output
    cv2.imshow(
        "MediaPipe 3D Face Transform",
        transformed_image
    )

    key = cv2.waitKey(1) & 0xFF

    # Press S to save
    if key == ord('s'):

        cv2.imwrite(
            output_path,
            transformed_image
        )

        print(
            f"Output saved to: {output_path}"
        )

    # Press Q to quit
    if key == ord('q'):
        break


# Release resources
cap.release()
face_mesh.close()
cv2.destroyAllWindows()