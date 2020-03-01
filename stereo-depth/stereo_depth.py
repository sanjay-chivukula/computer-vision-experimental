import numpy as np
import matplotlib.pyplot as plt
import cv2


class StereoDepth:
    WINDOW_SIZE = 3
    MINIMUM_DISPARITY = 16
    NUMBER_DISPARITY = 100 - MINIMUM_DISPARITY

    def __init__(self):
        self.stereo_compute_obj = cv2.StereoSGBM_create(
            minDisparity=StereoDepth.MINIMUM_DISPARITY,
            numDisparities=StereoDepth.NUMBER_DISPARITY,
            blockSize=16,
            P1=8 * 3 * StereoDepth.WINDOW_SIZE ** 2,
            P2=32 * 3 * StereoDepth.WINDOW_SIZE ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def compute_depth(self, image_l, image_r):
        disparity = self.stereo_compute_obj.compute(image_l, image_r)
        height, width = image_l.shape[:2]
        focal_length = 0.8 * width
        intrinsic = np.float32([[1, 0, 0, -0.5 * width],
                        [0, -1, 0, 0.5 * height],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1, 0]])
        points3d = cv2.reprojectImageTo3D(disparity, intrinsic)
        colors = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
        mask = disparity > disparity.min()

        cv2.imshow('left image', image_l)
        cv2.imshow('disparity', disparity)
        cv2.waitKey()

        out_points3d = points3d[mask]
        out_colors = colors[mask]
        return out_points3d, out_colors

def test_driver():
    depth_obj = StereoDepth()
    left = cv2.imread("left.png")
    left = cv2.resize(left, None, fx=0.2, fy=0.2)
    right = cv2.imread("right.png")
    right = cv2.resize(right, None, fx=0.2, fy=0.2)
    points3d, colors = depth_obj.compute_depth(left, right)
    for point3d, color in zip(points3d, colors):
        print(point3d, color)

if __name__ == "__main__":
    test_driver()