import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Output folder
output_folder = r"C:\Users\user\Documents\VS Code Work\Media_Pipe\outputs"
os.makedirs(output_folder, exist_ok=True)

# Output image
output_path = os.path.join(
    output_folder,
    "face_mesh_output.png"
)

# Drawing specification
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=1
)

# Open webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():

        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert BGR to RGB
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Process face
        results = face_mesh.process(image_rgb)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        # Draw face mesh
        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                # Face tessellation
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=
                    mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )

                # Face contours
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=
                    mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )

                # Iris landmarks
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=
                    mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                )

        # Mirror image
        display_image = cv2.flip(image, 1)

        # Show webcam
        cv2.imshow(
            "MediaPipe Face Mesh",
            display_image
        )

        key = cv2.waitKey(1) & 0xFF

        # Press S to save image
        if key == ord('s'):

            cv2.imwrite(
                output_path,
                display_image
            )

            print(
                f"Face mesh output saved to: {output_path}"
            )

        # Press Q to quit
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()