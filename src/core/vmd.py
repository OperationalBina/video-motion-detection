import imutils
import cv2
import numpy as np
from image_registration.core.registrar import Registrar


class VideoMotionDetector:
    def __init__(self, cap, history=100, min_size_for_movement=100, var_threshold=16, blurring_kernel_size=21,
                 display_width=500, equalize_hist=False, to_reg=True, ignore_border=30):
        self.cap = cap
        self.display_width = display_width
        self.display_height = 0
        self.blurring_kernel_size = blurring_kernel_size
        self.min_size_for_movement = min_size_for_movement
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history, var_threshold, False)
        self.equalize_hist = equalize_hist
        self.frame_counter = 0
        self.ignore_border = ignore_border
        self.history = history
        self.to_reg = to_reg
        self.anchor_image = None
        self.registrar = None
        self.last_mask = None

    def run_video_with_detections(self):
        ret, frame = self.cap.read()
        self.anchor_image = frame
        self.registrar = Registrar(point_count=500, top_matches_percent=0.5, is_only_affine=True)

        while ret:
            self.frame_counter += 1
            frame = imutils.resize(frame, width=self.display_width)
            self.display_height = frame.shape[0]

            if self.equalize_hist:
                frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
                frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

            if self.to_reg:
                frame = self.registrar.register_image_by_another_image(image_to_register=frame,
                                                                       anchor_image=self.anchor_image)

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

        frame_blurred = cv2.GaussianBlur(frame, (self.blurring_kernel_size, self.blurring_kernel_size), 0)

        fg_mask = self.back_sub.apply(frame_blurred)
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        fg_mask = cv2.erode(fg_mask, None)
        fg_mask = cv2.dilate(fg_mask, None, iterations=5)
        fg_morph_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        and_mask = cv2.bitwise_and(self.last_mask, fg_mask) if self.last_mask is not None else fg_mask
        and_mask = cv2.dilate(and_mask, None, iterations=1)
        self.last_mask = fg_mask

        contours, _ = cv2.findContours(and_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > self.min_size_for_movement] \
            if self.frame_counter >= self.history else []

        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if x > self.ignore_border and y > self.ignore_border and \
                    x + w < self.display_width - self.ignore_border and \
                    y + h < self.display_height - self.ignore_border:
                filtered_contours.append(contour)

        return frame, fg_morph_bgr, fg_mask_bgr, filtered_contours
