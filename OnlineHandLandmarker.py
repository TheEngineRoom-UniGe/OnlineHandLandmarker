import mediapipe as mp
import cv2
import time
import numpy as np

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
        self.landmark_names = {
            0: 'wrist',
            1: 'thumb_cmc', 2: 'thumb_mcp', 3: 'thumb_ip', 4: 'thumb_tip', 
            5: 'index_finger_mcp', 6: 'index_finger_pip', 7: 'index_finger_dip', 8: 'index_finger_tip',
            9: 'middle_finger_mcp', 10: 'middle_finger_pip', 11: 'middle_finger_dip', 12: 'middle_finger_tip',+
            13: 'ring_finger_mcp', 14: 'ring_finger_pip', 15: 'ring_finger_dip', 16: 'ring_finger_tip',
            17: 'pinky_mcp', 18: 'pinky_pip', 19: 'pinky_dip', 20: 'pinky_tip'
        }
        self.landmark_colors = {
            #color palm fuchsia
            0: (195, 47, 225), 1: (195, 47, 225), 5: (195, 47, 225), 9: (195, 47, 225), 13: (195, 47, 225), 17 : (195, 47, 225),
            #color thumb purple
            2: (118, 27, 137), 3: (118, 27, 137), 4: (118, 27, 137),
            #color index cyan
            6: (135, 198, 240), 7: (135, 198, 240), 8: (135, 198, 240),
            #color middle green
            10: (65, 243, 106), 11: (65, 243, 106), 12: (65, 243, 106),
            #color ring orange
            14: (226, 169, 54), 15: (226, 169, 54), 16: (226, 169, 54),
            #color pinkie red
            18: (246, 55, 55), 19: (246, 55, 55), 20: (246, 55, 55)
        }

        self.radius = 5
        self.cirlce_thickness = 2
        self.line_width = 5
        self.trail_length = 100
        self.trail = np.zeros((self.trail_length, 21, 2))
        self.trail[:] = np.nan
        self.alpha = 0.8


    def draw_landmarks_on_image(self, image, landmarks):
        original = image.copy()
        overlay = image.copy()
        self.trail = np.roll(self.trail, -1, 0)
        # print(len(landmarks.hand_landmarks))

        if len(landmarks.hand_landmarks) == 0:
            self.trail[-1, :, :] = np.nan

        for hand_landmarks in landmarks.hand_landmarks:

            for i, landmark in enumerate(hand_landmarks):
     
                cv2.circle(overlay, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), self.radius, self.landmark_colors[i], self.cirlce_thickness)
                self.trail[-1, i, :] = [int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])]

        delta = (self.alpha/2) / self.trail_length
        alpha = self.alpha/2

        overlay_cirlce = overlay.copy()

        gamma = (self.line_width / 2) / self.trail_length
        line_w = self.line_width / 2
        for i in range(self.trail.shape[0]-1):
            overlay = overlay_cirlce.copy()
            alpha += delta
            line_w += gamma
            if np.isnan(self.trail[i]).any() or np.isnan(self.trail[i+1]).any():
                break
            else:
                for j in range(self.trail.shape[1]):
                    cv2.line(overlay, (int(self.trail[i, j, 0]), int(self.trail[i, j, 1])),
                            (int(self.trail[i+1, j, 0]), int(self.trail[i+1, j, 1])), self.landmark_colors[j], int(line_w))
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        # image = cv2.addWeighted(image, self.alpha, original, 1-self.alpha, 0)
        return image


    def print_result(self, result: HandLandamarkerResult, output_image: mp.Image, timestamp_ms: int):
        # np_output_image = output_image.numpy_view()
        # print('hand landmarker result: {}'.format(result))
        self.annotated_image = self.draw_landmarks_on_image(cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB), result)
        # cv2.imshow('image', cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB))


    def start(self):

        vid = cv2.VideoCapture(0)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.print_result)

        with HandLandmarker.create_from_options(options) as landmarker:
            k = 0
            while(True):
                start = time.time()
                ret, frame = vid.read()
                frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


                # print(time.time())
                if k %2 == 0:
                    landmarker.detect_async(mp_image, self.stamp)
                if self.annotated_image is not None:
                    cv2.imshow('image', self.annotated_image)

                self.stamp += 1

                print(f'fps: {1/(time.time() - start)}')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':
    OHL = Online_Hand_Landmarker('/home/index1/Desktop/OnlineHandLandmarker/hand_landmarker.task')
    OHL.start()