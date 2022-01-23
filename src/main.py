import cv2

from src.core.vmd import vmd


def main():
    cap = cv2.VideoCapture('../data/road.mp4')
    vmd(cap=cap, frames_to_persist=10, min_size_for_movement=2000, movement_detected_persistence=100)


if __name__ == '__main__':
    main()
