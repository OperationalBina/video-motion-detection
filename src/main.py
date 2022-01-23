import cv2

from src.core.vmd import VideoMotionDetector


def main():
    cap = cv2.VideoCapture('../data/road.mp4')
    vmd = VideoMotionDetector(cap=cap)
    vmd.run_video_with_detections()


if __name__ == '__main__':
    main()
