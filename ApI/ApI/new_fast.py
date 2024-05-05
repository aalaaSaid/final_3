from fastapi import FastAPI, WebSocket
import cv2
import mediapipe as mp
import numpy as np
from starlette.websockets import WebSocketDisconnect
import uvicorn
import asyncio

app = FastAPI()

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global variables
update_width = 170
update_height = 50
glasses_images = []

# Load all sunglasses images
path = "C:/Users/007/final_3/ApI/ApI/"
num_images = 5  # Update this with the number of sunglasses images you have
for i in range(1, num_images + 1):
    image = cv2.imread(path + str(i) + ".png", cv2.IMREAD_UNCHANGED)
    if image is not None:
        glasses_images.append(cv2.resize(image, (update_width, update_height)))
    else:
        print(f"Error: Unable to load glasses image {i}.")

# Function to process each frame
def frame_generator(frame, glasses_image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image = frame
        if image is None:
            print("Error: Unable to read frame.")
            return

        imageHeight, imageWidth, _ = image.shape

        # Process the frame using face detection
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                # Extract the nose tip landmark
                normalizedLandmark = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                Nose_tip_x = pixelCoordinatesLandmark[0]  # NOSE
                Nose_tip_y = pixelCoordinatesLandmark[1]

                # Adjust the position based on the nose tip landmark
                y_adjust = int(update_height * 1.5)  # Adjust the vertical position
                x_adjust = int(update_width * 0.47)  # Center the sunglasses on the nose tip
                pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]

                # Create a mask for the sunglasses using the alpha channel
                _, _, _, alpha_channel = cv2.split(glasses_image)
                mask = alpha_channel

                # Overlay the sunglasses on the original image
                for c in range(0, 3):
                    image[pos[1]:pos[1] + update_height, pos[0]:pos[0] + update_width, c] = (
                            image[pos[1]:pos[1] + update_height, pos[0]:pos[0] + update_width, c] *
                            (1 - mask / 255.0) +
                            glasses_image[:, :, c] * (mask / 255.0)
                    )

        return image

# Function to read frames from the video source and emit them continuously
async def emit_frames(websocket: WebSocket, glasses_image_index: int):
    video_capture = cv2.VideoCapture(0)  # Change to your video source (e.g., video file path)
    if not video_capture.isOpened():
        print("Error: Failed to open video source.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame
        processed_frame = frame_generator(frame, glasses_images[glasses_image_index])

        # Emit the processed frame to the client
        await websocket.send_bytes(cv2.imencode('.jpg', processed_frame)[1].tobytes())

        await asyncio.sleep(0.03)  # Adjust the sleep time as needed to control frame rate

    video_capture.release()

@app.websocket("/camera_stream")
async def websocket_camera_stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        glasses_image_index = 0  # Initial image index
        while True:
            data = await websocket.receive_text()
            glasses_image_index = int(data)
            await emit_frames(websocket, glasses_image_index)
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
