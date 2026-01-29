import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision

model_path = r"data\face_landmarker.task"

# config 
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult
VisionRunningMode = vision.RunningMode

latest_result = None
# create face landmarker instance
def print_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    result_callback = print_result
)

landmarker = FaceLandmarker.create_from_options(options)

def process_webcam(landmarker):
    cap = cv2.VideoCapture(0)
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
            for face_landmarks in latest_result.face_landmarks:
                for landmark in face_landmarks:
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Face Mesh (Production)", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    process_webcam(landmarker)
