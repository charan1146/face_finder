from deepface import DeepFace
import cv2

# Your reference image
known_image = "charan.jpg"

# Open webcam
camera = cv2.VideoCapture(0)

while True:

    # Read frame from webcam
    success, frame = camera.read()

    # If camera fails
    if not success:
        print("Camera not working")
        break

    # Save current webcam frame
    cv2.imwrite("charan.jpg", frame)

    try:

        # Compare webcam face with your image
        result = DeepFace.verify(
            img1_path=known_image,
            img2_path="charan.jpg",
            detector_backend="opencv"
        )

        # If face matches
        if result["verified"]:
            text = "SRICHARAN"
        else:
            text = "Unknown"

    except Exception as e:

        print("ERROR:", e)
        text = "No Face"

    # Put text on webcam screen
    cv2.putText(
        frame,
        text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show webcam window
    cv2.imshow("AI Face Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
camera.release()

# Close all windows
cv2.destroyAllWindows()