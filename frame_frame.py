import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the sunglasses image (replace with your own image)
imgFront = cv2.imread("2D/Sun_Glasses/glass7.png", cv2.IMREAD_UNCHANGED)
if imgFront is None:
    print("Error in loading image")

# Resize glasses image
update_width = 150
update_height = 50
resize_glasses = cv2.resize(imgFront, (update_width, update_height))

s_h, s_w, _ = resize_glasses.shape

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    image = cv2.imread("2D/Eye_Glasses/1.jpg")
    imageHeight, imageWidth, _ = image.shape

    # Process the frame using face detection
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            # Extract the nose tip, left ear, and right ear landmarks

            # Extract nose landmark
            normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
            pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
            Nose_tip_x = pixelCoordinatesLandmark[0]  # NOSE
            Nose_tip_y = pixelCoordinatesLandmark[1]

            # Extract Left Ear coordinates
            normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
            pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
            Left_Ear_x = pixelCoordinatesLandmark[0]  # LEFT EAR
            Left_Ear_y = pixelCoordinatesLandmark[1]

            # Extract Right Ear coordinates
            normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
            pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

            Right_Ear_x = pixelCoordinatesLandmark[0]  # RIGHT EAR
            Right_Ear_y = pixelCoordinatesLandmark[1]

            # Calculate dimensions for the sunglasses
            # Initial size (adjust as needed)

            # ------------------------------------------------
            # Calculate the width of the face using the distance between the ears
            face_width = abs(Left_Ear_x - Right_Ear_x)
            sunglass_width = face_width  # No additional padding needed for automatic fitting

            # Calculate the height of the sunglasses to maintain aspect ratio
            sunglass_height = int((s_h / s_w) * sunglass_width)

            # Resize the sunglasses image
            glass_frame = cv2.resize(imgFront, (sunglass_width, sunglass_height))

            # Update 2
            # Calculate the new position with smoothing

            # Adjust the position based on the nose tip landmark
            y_adjust = int(sunglass_height * 1.3)  # Adjust the vertical position
            x_adjust = int(sunglass_width * 0.47)  # Center the sunglasses on the nose tip
            pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]

            pos = np.array([Nose_tip_x - x_adjust, Nose_tip_y - y_adjust])

            # Create a mask for the sunglasses using the alpha channel
            _, _, _, alpha_channel = cv2.split(glass_frame)
            mask = alpha_channel

            # Overlay the sunglasses on the original image
            for c in range(0, 3):
                image[pos[1]:pos[1] + sunglass_height, pos[0]:pos[0] + sunglass_width, c] = (
                        image[pos[1]:pos[1] + sunglass_height, pos[0]:pos[0] + sunglass_width, c] *
                        (1 - mask / 255.0) +
                        glass_frame[:, :, c] * (mask / 255.0)
                )

# Display the result
cv2.imshow("Sunglasses Overlay", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
