# flask api
from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np
import os
import imageio.v2 as imageio

def moving_average(new_pos, pos_buffer, buffer_size=5):
    pos_buffer.append(new_pos)
    if len(pos_buffer) > buffer_size:
        pos_buffer.pop(0)
    return np.mean(pos_buffer, axis=0)


app = Flask(__name__)

# Load the sunglasses image (replace with your own image)
imgFront = cv2.imread("glass_image/4.png", cv2.IMREAD_UNCHANGED)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# update
update_width = 150
update_hight = 50
# #resize glasses image
resize_glasses = cv2.resize(imgFront, (update_width, update_hight))

s_h, s_w, _ = resize_glasses.shape


@app.route('/process', methods=['GET', 'POST'])
def index():
    # Initialize a variable to store the previous frame's nose tip position
    prev_Nose_tip_x = None

    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({'error': 'no file part'})
        video = request.files['video']
        # intialize the video output path
        video_path = 'static/output.mp4'
        # video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for .mp4 files
        output_file = 'static/output.mp4'
        # vid_writer = cv2.VideoWriter(output_file, fourcc, 30.0, (506, 960))
        # vid_writer = cv2.VideoWriter('static/output.avi', -1, 20.0, (640, 480))
        # Initialize webcam capture
        input_video_path = os.path.join('static', video.filename)
        cap = cv2.VideoCapture(input_video_path)
        pos_buffer = []
        frame_count = 0
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                imageHeight, imageWidth, _ = image.shape

                # Process the frame using face detection
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.detections:
                    for detection in results.detections:
                        # Extract the nose tip, left ear, and right ear landmarks

                        # Extract nose landmark
                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.NOSE_TIP)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth,
                                                                                               imageHeight)
                        Nose_tip_x = pixelCoordinatesLandmark[0]  # NOSE
                        Nose_tip_y = pixelCoordinatesLandmark[1]

                        # Extract Left Ear coordinates

                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth,
                                                                                               imageHeight)
                        Left_Ear_x = pixelCoordinatesLandmark[0]  # LEFT EAR
                        Left_Ear_y = pixelCoordinatesLandmark[1]

                        # Extract  Right Ear coordinates
                        normalizedLandmark = mp_face_detection.get_key_point(detection,
                                                                             mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                               normalizedLandmark.y,
                                                                                               imageWidth,
                                                                                               imageHeight)

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
                        glass_frame = imgFront.copy()
                        glass_frame = cv2.resize(glass_frame, (sunglass_width, sunglass_height))

                        # update 2
                        # calculate the new position with smoothing

                        # Adjust the position based on the nose tip landmark
                        y_adjust = int(sunglass_height * 1.3)  # Adjust the vertical position
                        x_adjust = int(sunglass_width * 0.47)  # Center the sunglasses on the nose tip
                        pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]

                        new_glasses_pos = np.array([Nose_tip_x - x_adjust, Nose_tip_y - y_adjust])

                        # update 2
                        # new_position is stored in new glasses position

                        # calculate the stabilized position
                        stabilized_pos = moving_average(new_glasses_pos, pos_buffer).astype(int)

                        # Check if previous nose tip position is available
                        if prev_Nose_tip_x is not None:
                            # Calculate movement offset based on nose tip position change
                            movement_offset_x = Nose_tip_x - prev_Nose_tip_x

                            # Update the position of the sunglasses based on movement
                            pos[0] += movement_offset_x

                        # Update previous nose tip position

                        prev_Nose_tip_x = Nose_tip_x

                        # ----------------------------------------------

                        # Create a mask for the sunglasses
                        *_, mask = cv2.split(glass_frame)
                        # print(cv2.split(imgFront))
                        maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                        # print(imgFront.shape)
                        # print(maskBGRA.shape)

                        imgRGBA = cv2.bitwise_and(glass_frame, maskBGRA)
                        imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

                        # Overlay the sunglasses on the original image
                        imgMaskFull = np.zeros_like(image, np.uint8)
                        imgMaskFull[stabilized_pos[1]:stabilized_pos[1] + sunglass_height,
                        stabilized_pos[0]:stabilized_pos[0] + sunglass_width, :] = imgRGB
                        imgMaskFull2 = np.ones_like(image, np.uint8) * 255
                        maskBGRInv = cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                        imgMaskFull2[stabilized_pos[1]:stabilized_pos[1] + sunglass_height,
                        stabilized_pos[0]:stabilized_pos[0] + sunglass_width, :] = maskBGRInv

                        image = cv2.bitwise_and(image, imgMaskFull2)
                        image = cv2.bitwise_or(image, imgMaskFull)
                cv2.imwrite(f"static/images/frame_{frame_count}.jpg", image)
                frame_count += 1
        cap.release()
        cv2.destroyAllWindows()
    return jsonify({"success": "no error"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
