import cv2

from src.core.vmd import VideoMotionDetector


def main():
    cap = cv2.VideoCapture('../data/vis_87.mp4')
    vmd = VideoMotionDetector(cap=cap, to_reg=True)
    vmd.run_video_with_detections()


if __name__ == '__main__':
    main()
