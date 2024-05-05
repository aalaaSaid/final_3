import cv2
import mediapipe as mp
import numpy as np
import keyboard

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the sunglasses image (replace with your own image)
imgFront = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
s_h, s_w, _ = imgFront.shape

# Initialize webcam capture
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        
        imageHeight,imageWidth,_ = image.shape

        # Process the frame using face detection
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                # Extract the nose tip, left ear, and right ear landmarks
                
                # Extract nose landmark
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                Nose_tip_x = pixelCoordinatesLandmark[0]     # NOSE    
                Nose_tip_y = pixelCoordinatesLandmark[1]
                
                # Extract Left Ear coordinates
                
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                Left_Ear_x = pixelCoordinatesLandmark[0]     # LEFT EAR      
                Left_Ear_y = pixelCoordinatesLandmark[1]
                
                # Extract  Right Ear coordinates
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                
                Right_Ear_x = pixelCoordinatesLandmark[0]    # RIGHT EAR      
                Right_Ear_y = pixelCoordinatesLandmark[1]
                
                # Extract Left Eye coordinates
                '''
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_image_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                
                Left_EYE_x = pixelCoordinatesLandmark[0]     # LEFT EYE    
                Left_EYE_y = pixelCoordinatesLandmark[1]
                
                # Extract Right Eye coordinates 
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_image_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                
                Right_EYE_x = pixelCoordinatesLandmark[0]    # RIGHT EYE    
                Right_EYE_y = pixelCoordinatesLandmark[1]
                '''
                # Calculate dimensions for the sunglasses
                # Initial size (adjust as needed)
        
                 
                sunglass_width = Left_Ear_x - Right_Ear_x + 60
                sunglass_height = int((s_h / s_w) * sunglass_width)
                

                
                imgFront = cv2.resize(imgFront, (sunglass_width, sunglass_height), None, 0.1, 0.1)
                
                

                # Adjust position based on nose tip landmark
                y_adjust = int((sunglass_height /50) * 50) # Fine-tune this value
               #x_adjust = int((sunglass_width / 194) * 100)
                x_adjust = int((sunglass_width*0.5))
                pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]

                # Create a mask for the sunglasses
                *_, mask = cv2.split(imgFront)
                maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
                imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

                # Overlay the sunglasses on the original image
                imgMaskFull = np.zeros_like(image, np.uint8)
                imgMaskFull[pos[1]:pos[1] + sunglass_height, pos[0]:pos[0] + sunglass_width, :] = imgRGB
                imgMaskFull2 = np.ones_like(image, np.uint8) * 255
                maskBGRInv = cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                imgMaskFull2[pos[1]:pos[1] + sunglass_height, pos[0]:pos[0] + sunglass_width, :] = maskBGRInv

                image = cv2.bitwise_and(image, imgMaskFull2)
                image = cv2.bitwise_or(image, imgMaskFull)

        # Display the result
        cv2.imshow('Sunglass Effect', image)

        if keyboard.is_pressed('q'):
            break

        cv2.waitKey(5)

cap.release()
cv2.destroyAllWindows()
