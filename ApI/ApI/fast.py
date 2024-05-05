from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
from starlette.websockets import WebSocketDisconnect
import uvicorn
import asyncio

app = FastAPI()

# Serve static files
app.mount("/API", StaticFiles(directory="API"), name="API")

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# update
update_width = 170
update_height = 50
# D:/gradution/final_3/ApI/ApI/8.png

# Load the sunglasses image (replace with your own image)
# C:\Users\007\final_3\ApI\ApI\4.png
path="D:/gradution/final_3/ApI/ApI/"
pic_num=1
glasses_image = cv2.imread(path +str(pic_num)+".png", cv2.IMREAD_UNCHANGED)
imgFront=glasses_image
resize_glasses = cv2.resize(imgFront, (update_width,update_height ))
s_h, s_w, _ = resize_glasses.shape
if glasses_image is None:
    print("Error: Unable to load glasses image.")
    exit()

# Function to process each frame
def frame_generator(frame):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image = frame
        if image is None:
            print("Error: Unable to read frame.")
            return

        imageHeight, imageWidth, _ = image.shape

        # Resize the glasses image
        resize_glasses = cv2.resize(glasses_image, (update_width, update_height))

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

                # Adjust the position based on the nose tip landmark
                y_adjust = int(update_height * 1.5)  # Adjust the vertical position
                x_adjust = int(update_width * 0.47)  # Center the sunglasses on the nose tip
                pos = [Nose_tip_x - x_adjust, Nose_tip_y - y_adjust]

                # Create a mask for the sunglasses using the alpha channel
                _, _, _, alpha_channel = cv2.split(resize_glasses)
                mask = alpha_channel

                # Overlay the sunglasses on the original image
                for c in range(0, 3):
                    image[pos[1]:pos[1] + update_height, pos[0]:pos[0] + update_width, c] = (
                            image[pos[1]:pos[1] + update_height, pos[0]:pos[0] + update_width, c] *
                            (1 - mask / 255.0) +
                            resize_glasses[:, :, c] * (mask / 255.0)
                    )

        return image

# Function to read frames from the video source and emit them continuously
async def emit_frames(websocket: WebSocket):
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
        processed_frame = frame_generator(frame)

        # Emit the processed frame to the client
        await websocket.send_bytes(cv2.imencode('.jpg', processed_frame)[1].tobytes())

        await asyncio.sleep(0.03)  # Adjust the sleep time as needed to control frame rate

    video_capture.release()

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        # Start emitting frames when a client connects
        await emit_frames(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/")
async def get():
    content = """
    <html>
        <head>
            <title>Sunglasses Filter</title>
        </head>
        <body>
            <h1>Sunglasses Filter</h1>
            <img id="bgImage" src="" width="640" height="480">
            <script>
                var ws = new WebSocket("ws://localhost:8000/");
                var img = document.getElementById("bgImage");
                ws.binaryType = "arraybuffer";
                ws.onmessage = function(event) {
                    var arrayBufferView = new Uint8Array(event.data);
                    var blob = new Blob([arrayBufferView], { type: "image/jpeg" });
                    var imageUrl = URL.createObjectURL(blob);
                    img.src = imageUrl;
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
