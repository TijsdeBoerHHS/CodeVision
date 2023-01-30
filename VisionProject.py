import cv2
import numpy as np

cam = cv2.VideoCapture(0)


def conversion(x, y):
    xrot = 0
    yrot = 0
    xfin = 0
    yfin = 0
    rotation = 35*(np.pi/180)  # 35 Degrees rotation in radians
    rot = np.array(([np.cos(rotation), np.sin(rotation)],
                   [-np.sin(rotation), np.cos(rotation)]))
    xy = np.array([x, y])
    xrot, yrot = np.dot(rot, xy)
    xfin = 0  # scaling
    yfin = 0  # scaling
    return (xfin, yfin)


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


def get_color_contours(frame, color_lower, color_upper, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)

    color_mask = cv2.inRange(hsv, color_lower, color_upper)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # TODO: variable is never used
    # segmented_color = cv2.bitwise_and(frame, frame, mask=color_mask)

    # color_contours, color_hierarchy
    color_contours, _ = cv2.findContours(
        color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_color = cv2.drawContours(frame, color_contours, -1, color, 3)
    return color_contours, output_color


def get_contour_coordinate(color_contours):
    if len(color_contours):
        # TODO: what is M?
        M = cv2.moments(color_contours[0])
        if M['m00'] != 0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            return {'x': x, 'y': y}

    return None


def draw_contour_and_text(frame, color_contours, coordinate):
    cv2.drawContours(frame, [color_contours[0]], -1, (0, 255, 0), 2)
    cv2.circle(frame, (coordinate['x'], coordinate['y']), 7, (0, 0, 255), -1)
    cv2.putText(frame, "center", (coordinate['x'] - 20, coordinate['y'] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    print(f"x: {coordinate['x']} y: {coordinate['y']}")


def get_color_coordinate(frame, color_contours, output_color):
    coordinate = get_contour_coordinate(color_contours)
    if coordinate:
        draw_contour_and_text(frame, color_contours, coordinate)
        frame, angle = get_rotation_angle(frame, color_contours)

        return [True, coordinate['x'], coordinate['y'], output_color, angle]

    return [False, 0, 0, frame, 0]


def get_blue_coordinate(frame):
    color_lower = np.array([107, 156, 130])
    color_upper = np.array([129, 241, 255])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (255, 0, 0))

    return get_color_coordinate(frame, color_contours, output_color)


def get_red_coordinate(frame):
    color_lower = np.array([0, 108, 159])
    color_upper = np.array([179, 170, 186])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 0, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_yellow_coordinate(frame):
    color_lower = np.array([20, 185, 185])
    color_upper = np.array([34, 255, 255])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 255, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_green_coordinate(frame):
    red_lower = np.array([65, 77, 83])
    red_upper = np.array([93, 145, 121])

    color_contours, output_color = get_color_contours(frame, red_lower, red_upper, (0, 255, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_orange_coordinate(frame):
    color_lower = np.array([6, 0, 160])
    color_upper = np.array([14, 159, 240])
    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 165, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_white_coordinate(frame):
    color_lower = np.array([0, 0, 210])
    color_upper = np.array([180, 40, 255])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 165, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_purple_coordinate(frame):
    color_lower = np.array([88, 20, 0])
    color_upper = np.array([131, 126, 169])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 165, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def getframe():
    _, frame = cam.read()
    return frame


def getnextframe(colors, frame):
    detect = False
    if (colors[0] == True) and (detect == False):
        detect, x, y, newframe, angle = get_blue_coordinate(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "b", rotangle)
    if (colors[1] == True) and (detect == False):
        detect, x, y, newframe, angle = get_red_coordinate(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "r", rotangle)
    if (colors[2] == True) and (detect == False):
        detect, x, y, newframe, angle = get_yellow_coordinate(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "y", rotangle)
    if (colors[3] == True) and (detect == False):
        detect, x, y, newframe, angle = get_green_coordinate(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "g", rotangle)
    if (colors[4] == True) and (detect == False):
        detect, x, y, newframe, angle = get_orange_coordinate(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "o", rotangle)
    if (colors[5] == True) and (detect == False):
        detect, x, y, newframe, angle = get_white_coordinate(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "w", rotangle)
    if (colors[6] == True) and (detect == False):
        detect, x, y, newframe, angle = get_purple_coordinate(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "p", rotangle)
    if all(items is False for items in colors) == False:
        if detect == False:
            print("Piece Error")
            # PE stands for Piece Error, which means no needed pieces found
            return (frame, 0, 0, "PE", 0)
    if all(items is False for items in colors) == True:
        print('All Pieces fitted')
        return (frame, 0, 0, "SO", 0)  # SO stands for Solved


def checkmsg(msg, colors):
    if msg == "b":
        colors[0] = False
        print("Blue")
        return (colors)
    elif msg == "r":
        colors[1] = False
        print("Red")
        return (colors)
    elif msg == "y":
        colors[2] = False
        print("Yellow")
        return (colors)
    elif msg == "g":
        colors[3] = False
        print("Green")
        return (colors)
    elif msg == "o":
        colors[4] = False
        print("Orange")
        return (colors)
    elif msg == "w":
        colors[5] = False
        print("White")
        return (colors)
    elif msg == "p":
        colors[6] = False
        print("Purple")
        return (colors)
    elif msg == "PE":
        print("PE")
        # send message to plc
        pass
    return (colors)


def main():
    frame = getframe()
    cv2.imshow("Detection", frame)
    blue = True
    red = True
    yellow = True
    green = True
    orange = True
    white = True
    purple = True
    colors = [blue, red, yellow, green, orange, white, purple]
    while (True):

        if cv2.waitKey(0) == ord("n"):
            frame = getframe()
            if all(items is False for items in colors) == True:
                print('All Pieces solved')
                for i in range(0, 7):
                    colors[i] = True
            frame, x, y, msg, rotangle = getnextframe(colors, frame)
            colors = checkmsg(msg, colors)
            #xfin,yfin = conversion(x,y)
            cv2.imshow("Detection", frame)


if __name__ == "__main__":
    main()
