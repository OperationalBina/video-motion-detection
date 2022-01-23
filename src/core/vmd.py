import imutils
import cv2
import numpy as np


class VideoMotionDetector:
    def __init__(self, cap, history=100, min_size_for_movement=100, var_threshold=16, bluring_kernel_size=21,
                 display_width=500):
        self.cap = cap
        self.display_width = display_width
        self.bluring_kernel_size = bluring_kernel_size
        self.min_size_for_movement = min_size_for_movement
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history, var_threshold, False)

    def run_video_with_detections(self):
        ret, frame = self.cap.read()

        while ret:
            frame = imutils.resize(frame, width=self.display_width)
            frame, thresh_bgr, fg_mask_bgr, contours = self.get_detections(frame)
            frame = self.draw_contours_on_image(image=frame, contours=contours)

            display = np.hstack((thresh_bgr, frame, fg_mask_bgr))
            cv2.imshow("frame", display)

            ch = cv2.waitKey(1)
            if ch & 0xFF == ord('q'):
                break

            ret, frame = self.cap.read()

        cv2.destroyAllWindows()
        self.cap.release()

    @staticmethod
    def draw_contours_on_image(image, contours, color=(0, 255, 0)):
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        return image

    def get_detections(self, frame):
        frame_blurred = cv2.GaussianBlur(frame, (self.bluring_kernel_size, self.bluring_kernel_size), 0)

        fg_mask = self.back_sub.apply(frame_blurred)
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        thresh = cv2.erode(fg_mask, None)
        thresh = cv2.dilate(thresh, None, iterations=5)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > self.min_size_for_movement]

        return frame, thresh_bgr, fg_mask_bgr, contours
