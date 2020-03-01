import os

import numpy as np
import matplotlib.pyplot as plt
import cv2


def feature_detect(video_path_input: str, video_path_output: str) -> None:
    detection_obj = FeatureDetect(video_path_input, video_path_output)
    detection_obj.detect()
    detection_obj.release()


class FeatureDetect:
    def __init__(self, input_video_path: str, output_video_path: str) -> None:
        self.input_video_path: str = self._verify_video_path(input_video_path)
        self.video_reader_object: cv2.VideoCapture = cv2.VideoCapture(self.input_video_path)

        self.output_video_path: str = output_video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer_object: cv2.VideoWriter = cv2.VideoWriter(self.output_video_path, self.fourcc, 20.0,
                                                                    (640, 480))

    def detect(self):
        orb_detector = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
        while self.video_reader_object.isOpened():
            ret, frame = self.video_reader_object.read()
            if not ret:
                break

            # preprocessing
            frame = cv2.resize(frame, (640, 480))

            # key point detection
            key_points = orb_detector.detect(frame, None)
            key_points, descriptors = orb_detector.compute(frame, key_points)
            output_frame = cv2.drawKeypoints(frame, key_points, None, color=(0, 0, 255), flags=0)

            # writing output_frame to video file
            self.video_writer_object.write(output_frame)

            # displaying output_frame
            cv2.imshow("Feature Detection", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release(self):
        self.video_reader_object.release()
        self.video_writer_object.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _verify_video_path(video_path: str) -> str:
        if not video_path:
            raise ValueError("Empty video path string!!")
        if not os.path.isfile(os.path.abspath(video_path)):
            raise FileNotFoundError("Path to input video is incorrect or file does not exist!!")
        return video_path


def test_driver():
    input_path = "test.mp4"
    output_path = "out.avi"
    feature_detect(input_path, output_path)


if __name__ == "__main__":
    test_driver()
