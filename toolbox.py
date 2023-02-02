import cv2
import numpy as np
from vision import get_color_contours

"""https://stackoverflow.com/a/57474183/10598904"""


def find_color_threshold(att_img, mode=None):
    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('HMin', 'image', 0, 179, lambda _: None)
    cv2.createTrackbar('SMin', 'image', 0, 255, lambda _: None)
    cv2.createTrackbar('VMin', 'image', 0, 255, lambda _: None)
    cv2.createTrackbar('HMax', 'image', 0, 179, lambda _: None)
    cv2.createTrackbar('SMax', 'image', 0, 255, lambda _: None)
    cv2.createTrackbar('VMax', 'image', 0, 255, lambda _: None)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0
    img = att_img
    waitTime = 33

    while (1):
        if callable(att_img):
            img = att_img()

        resizeFactor = img.shape[0] / 600
        resizeShape = (round(img.shape[1] / resizeFactor), round(img.shape[0] / resizeFactor))

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')

        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        if mode == 1:
            _, output, _ = get_color_contours(img, lower, upper, (255, 255, 255))
        if mode == 2:
            _, _, output = get_color_contours(img, lower, upper, (255, 255, 255))
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            output = cv2.bitwise_and(img, img, mask=mask)

        # Print if there is a change in HSV value
        if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
            print(f"[{hMin}, {sMin}, {vMin}] [{hMax}, {sMax}, {vMax}]")
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image', cv2.resize(output, resizeShape))

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
