import cv2
import time
import mediapipe as mp
from collections import deque
from mediapipe.tasks.python import vision

model_path = r"data\face_landmarker.task"

# config 
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult
VisionRunningMode = vision.RunningMode

latest_result = None
WINDOW_SIZE = 30
frame_buffer = deque(maxlen=WINDOW_SIZE)

# create face landmarker instance
def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,             
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    result_callback=result_callback
)

landmarker = FaceLandmarker.create_from_options(options)

def process_webcam(landmarker):
    cap = cv2.VideoCapture(0)
    #reduce cpu load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        timestamp_ms = int((time.time() - start_time) * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw face mesh if result exists
        if latest_result and latest_result.face_landmarks:
            frame_data = {
                "timestamp": timestamp_ms,
                "landmarks": latest_result.face_landmarks[0],
                "blendshapes": (
                    latest_result.face_blendshapes[0]
                    if latest_result.face_blendshapes else None
                ),
                "pose": (
                    latest_result.facial_transformation_matrixes[0]
                    if latest_result.facial_transformation_matrixes else None
                ),
            }

            frame_buffer.append(frame_data) # append frame data to deque
            if len(frame_buffer) == WINDOW_SIZE:
                print("Temporal window obtained.") 

        cv2.imshow("Face Mesh (Production)", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    process_webcam(landmarker)
