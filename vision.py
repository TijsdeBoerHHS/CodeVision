import cv2
import numpy as np


def get_coordinate_factory(attributes):
    color_lower = np.array(attributes['lower'])
    color_upper = np.array(attributes['upper'])

    def closure_function(frame):
        color_contours, output_color, _ = get_color_contours(
            frame, color_lower, color_upper, attributes['colors']['contour'])
        return get_color_coordinate(frame, color_contours, output_color, attributes['colors'])

    return closure_function


def get_rotation_angle(frame, contours):
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        if area < 500 or 10000 > area:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        angle = 90 - angle if width < height else -angle

        label = "  Rotation Angle: " + str(angle) + " degrees"
        cv2.putText(frame, label, (center[0]-50, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        print(angle)
        return frame, angle

    return frame, 0


def contour_fiter(color_contours):
    color_contours = list(color_contours)
    if not color_contours:
        return color_contours

    c_max = 0
    c_index = None
    for c in range(len(color_contours)):
        area = cv2.contourArea(color_contours[c])
        if area > c_max:
            c_max = area
            c_index = c

    color_contours = [color_contours[c_index]]
    return color_contours


def get_color_contours(frame, color_lower, color_upper, contour_color, kernel=(7, 7)):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones(kernel, np.uint8)

    color_mask = cv2.inRange(hsv, color_lower, color_upper)
    # cv2.imwrite('debug/color_mask1.png', color_mask)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    # cv2.imwrite('debug/color_mask3.png', color_mask)
    color_contours, _ = cv2.findContours(
        color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_color = cv2.drawContours(frame, color_contours, -1, contour_color, 3)
    color_contours = contour_fiter(color_contours)
    return color_contours, output_color, color_mask


def get_contour_coordinate(color_contours):
    if len(color_contours):
        M = cv2.moments(color_contours[0])
        if M['m00'] != 0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            return {'x': x, 'y': y}

    return None


def draw_contour_and_text(frame, color_contours, coordinate, colors):
    cv2.drawContours(frame, [color_contours[0]], -1, colors['contour'], 2)
    cv2.circle(frame, (coordinate['x'], coordinate['y']), 7, colors['center'], -1)
    cv2.putText(frame, "center", (coordinate['x'] - 20, coordinate['y'] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 2)
    print(f"x: {coordinate['x']} y: {coordinate['y']}")


def get_color_coordinate(frame, color_contours, output_color, colors=None):
    if not colors:
        colors = {
            'contour': (0, 0, 0),
            'text': (0, 0, 0),
            'center': (0, 0, 255),
        }

    coordinate = get_contour_coordinate(color_contours)
    if coordinate:
        draw_contour_and_text(frame, color_contours, coordinate, colors)
        frame, angle = get_rotation_angle(frame, color_contours)

        return [True, coordinate['x'], coordinate['y'], output_color, angle]

    return [False, 0, 0, frame, 0]
