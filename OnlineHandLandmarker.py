import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandamarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class Online_Hand_Landmarker():
    def __init__(self, model_path):
        self.annotated_image = None
        self.stamp = 0
        self.model_path = model_path

    def draw_landmarks_on_image(self, image, landmarks):
        # print(len(landmarks.hand_landmarks))
        for hand_landmarks in landmarks.hand_landmarks:
            for landmark in hand_landmarks:
                print(landmark.x)
                print(image.shape)
                print((int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])))
        #     print(landmark)
                cv2.circle(image, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 10, (255, 0, 0), 5)
        return image

    def print_result(self, result: HandLandamarkerResult, output_image: mp.Image, timestamp_ms: int):
        # np_output_image = output_image.numpy_view()
        print('hand landmarker result: {}'.format(result))
        self.annotated_image = self.draw_landmarks_on_image(cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB), result)
        # cv2.imshow('image', cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB))


    def start(self):

        vid = cv2.VideoCapture(0)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result)

        with HandLandmarker.create_from_options(options) as landmarker:
            while(True):
                ret, frame = vid.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


                # print(time.time())
                landmarker.detect_async(mp_image, self.stamp)
                if self.annotated_image is not None:
                    cv2.imshow('image', self.annotated_image)

                self.stamp += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':
    OHL = Online_Hand_Landmarker('./hand_landmarker.task')
    OHL.start()