import cv2
import mediapipe as mp
import os

# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def main():

    # Output folder
    output_folder = r"C:\Users\user\Documents\VS Code Work\Media_Pipe\outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Output image
    output_path = os.path.join(
        output_folder,
        "instant_motion_tracking_output.png"
    )

    # Set up video capture
    cap = cv2.VideoCapture(0)

    # Set up MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():

            # Read frame
            success, frame = cap.read()

            if not success:
                print("Could not read webcam.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Process with MediaPipe
            results = hands.process(image_rgb)

            # Draw hand landmarks
            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # Show live camera
            cv2.imshow(
                "Instant Motion Tracking",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Press S to save output image
            if key == ord('s'):

                cv2.imwrite(
                    output_path,
                    frame
                )

                print(
                    f"Output saved to: {output_path}"
                )

            # Press Q to quit
            if key == ord('q'):
                break

    # Release webcam
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()