# flask api
from flask import Flask, request, Response
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS


def moving_average(new_pos, pos_buffer, buffer_size=5):
    pos_buffer.append(new_pos)
    if len(pos_buffer) > buffer_size:
        pos_buffer.pop(0)
    return np.mean(pos_buffer, axis=0)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


socketio = SocketIO(app)

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# update
update_width = 150
update_hight = 50


# #resize glasses image 
resize_glasses=cv2.resize(imgFront,(update_width,update_hight))

s_h, s_w, _ = resize_glasses.shape

@socketio.on('video_frame')
def handle_video_frame(frame, img_num):
    # Load the sunglasses image (replace with your own image)
    glasses_image = cv2.imread(str(img_num)+".png", cv2.IMREAD_UNCHANGED)
        # #resize glasses image 
    resize_glasses=cv2.resize(glasses_image,(update_width,update_hight))

    s_h, s_w, _ = resize_glasses.shape
    
    # Process the received frame here
    # print("Received frame:", frame[:50])
    # # add your code here
    # print("Image number:", img_num)

    def frame_generator(frame):
        frame=cv2.imread(frame)
        #initiallize buffer for the glasses positions 
        pos_buffer=[]
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            
                imageHeight,imageWidth,_ = frame.shape

                # Process the frame using face detection
                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
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
                        
                        # Calculate dimensions for the sunglasses
                        # Initial size (adjust as needed)
                        
                        

                        # ------------------------------------------------
                        # Calculate the width of the face using the distance between the ears
                        face_width = abs(Left_Ear_x - Right_Ear_x)
                        sunglass_width = face_width  # No additional padding needed for automatic fitting
                        
                        # Calculate the height of the sunglasses to maintain aspect ratio
                        sunglass_height = int((s_h / s_w) * sunglass_width)
                        
                        # Resize the sunglasses image
                        glass_frame = glasses_image.copy()
                        glass_frame = cv2.resize(glass_frame, (sunglass_width, sunglass_height))
                        
                        # update 2
                        #caculate the new position with smoothing 
                        
                        # Adjust the position based on the nose tip landmark
                        y_adjust = int(sunglass_height * 1.3)  # Adjust the vertical position
                        x_adjust = int(sunglass_width * 0.5)   # Center the sunglasses on the nose tip
                        pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]
                        
                        new_glasses_pos=np.array([Nose_tip_x - x_adjust,Nose_tip_y - y_adjust])
                        
                        # update 2
                        #new_position is stored in new glasses position 
                        
                        #calculate the stabilized position 
                        stabilized_pos= moving_average(new_glasses_pos,pos_buffer).astype(int)
                        
                        # Check if previous nose tip position is available
                        if prev_Nose_tip_x is not None:
                        # Calculate movement offset based on nose tip position change
                            movement_offset_x = Nose_tip_x - prev_Nose_tip_x

                        # Update the position of the sunglasses based on movement
                            pos[0] += movement_offset_x

                        # Update previous nose tip position
                        
                        prev_Nose_tip_x = Nose_tip_x

                        #----------------------------------------------

                        # Create a mask for the sunglasses
                        *_, mask = cv2.split(glass_frame)
                        # print(cv2.split(imgFront))
                        maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                        # print(imgFront.shape)
                        # print(maskBGRA.shape)
                        
                        imgRGBA = cv2.bitwise_and(glass_frame, maskBGRA)
                        imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

                        # Overlay the sunglasses on the original image
                        imgMaskFull = np.zeros_like(frame, np.uint8)
                        imgMaskFull[stabilized_pos[1]:stabilized_pos[1] + sunglass_height,stabilized_pos[0]:stabilized_pos[0] + sunglass_width, :] = imgRGB
                        imgMaskFull2 = np.ones_like(frame, np.uint8) * 255
                        maskBGRInv = cv2.bitwise_not(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                        imgMaskFull2[stabilized_pos[1]:stabilized_pos[1] + sunglass_height, stabilized_pos[0]:stabilized_pos[0] + sunglass_width, :] = maskBGRInv

                        frame = cv2.bitwise_and(frame, imgMaskFull2)
                        frame = cv2.bitwise_or(frame, imgMaskFull)
                return frame
                    

    frame_data = frame_generator(frame=frame)

    socketio.emit("generated_frame", frame_data)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    socketio.run(app, debug=True)
