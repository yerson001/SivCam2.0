# import the necessary packages
import numpy as np
import imutils
import cv2


class MotionDetector:
    def __init__(self, accumWeight=0.5):
        self.accumWeight = accumWeight
        self.bg = None

    def update(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
        #eliminar manchas
        #diff = cv2.absdiff(frame,frame1)
        # gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray,(5,5),0)
        # _,thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
        # dilated = cv2.dilate(thresh,None,iterations=3)

        thresh = cv2.dilate(thresh, None, iterations=5)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)
        if len(cnts) == 0:
            return delta * 10, None

        detected_boxes = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            detected_boxes.append([x, y, x + w, y + h])
        return (delta * 10, detected_boxes)
