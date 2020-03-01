import os

import numpy as np
import matplotlib.pyplot as plt
import cv2


def feature_match(video_path_input: str, video_path_output: str) -> None:
    detection_obj = FeatureMatch(video_path_input, video_path_output)
    detection_obj.detect_and_match()
    detection_obj.release()


class FeatureMatch:
    def __init__(self, input_video_path: str, output_video_path: str) -> None:
        self.input_video_path: str = self._verify_video_path(input_video_path)
        self.video_reader_object: cv2.VideoCapture = cv2.VideoCapture(self.input_video_path)

        self.output_video_path: str = output_video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer_object: cv2.VideoWriter = cv2.VideoWriter(self.output_video_path, self.fourcc, 20.0,
                                                                    (640, 480))

    def detect_and_match(self):
        orb_detector = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
        brute_force_matcher_obj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        prev_frame, prev_key_points, prev_descriptors = None, None, None
        while self.video_reader_object.isOpened():
            ret, curr_frame = self.video_reader_object.read()
            if not ret:
                break

            # preprocessing
            curr_frame = cv2.resize(curr_frame, (640, 480))

            # key point detection and matching
            curr_key_points = orb_detector.detect(curr_frame, None)
            curr_key_points, curr_descriptors = orb_detector.compute(curr_frame, curr_key_points)
            if prev_key_points is None or prev_descriptors is None or prev_frame is None:
                prev_frame, prev_key_points, prev_descriptors = curr_frame, curr_key_points, curr_descriptors
                continue

            matches = brute_force_matcher_obj.match(prev_descriptors, curr_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # drawing matches
            matches_frame = cv2.drawMatches(prev_frame, prev_key_points, curr_frame, curr_key_points, matches, None, flags=2)

            # writing output_frame to video file
            self.video_writer_object.write(matches_frame)

            # displaying output_frame
            cv2.imshow("Feature Detection", matches_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_frame, prev_key_points, prev_descriptors = curr_frame, curr_key_points, curr_descriptors

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
    feature_match(input_path, output_path)


if __name__ == "__main__":
    test_driver()
