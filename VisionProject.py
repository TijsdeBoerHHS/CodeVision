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


def getrotangle(frame, contours):

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

        if width < height:
            angle = 90 - angle
        else:
            angle = -angle

        label = "  Rotation Angle: " + str(angle) + " degrees"
        cv2.putText(frame, label, (center[0]-50, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        print(angle)
        return frame, angle
    return frame, 0


def getbluecoord(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    blue_lower = np.array([107, 156, 130])
    blue_upper = np.array([129, 241, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    segmented_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    blue_contours, blue_hierarchy = cv2.findContours(
        blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_blue = cv2.drawContours(frame, blue_contours, -1, (255, 0, 0), 3)
    if len(blue_contours):
        M = cv2.moments(blue_contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [blue_contours[0]], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            frame, angle = getrotangle(frame, blue_contours)
            return [True, cx, cy, output_blue, angle]
    else:
        return [False, 0, 0, frame, 0]


def getredcoord(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    red_lower = np.array([0, 108, 159])
    red_upper = np.array([179, 170, 186])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    segmented_red = cv2.bitwise_and(frame, frame, mask=red_mask)
    red_contours, red_hierarchy = cv2.findContours(
        red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_red = cv2.drawContours(frame, red_contours, -1, (0, 0, 255), 3)
    if len(red_contours):
        M = cv2.moments(red_contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [red_contours[0]], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            frame, angle = getrotangle(frame, red_contours)
            return [True, cx, cy, output_red, angle]
    else:
        return [False, 0, 0, frame, 0]


def getyellowcoord(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    yellow_lower = np.array([20, 185, 185])
    yellow_upper = np.array([34, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    segmented_yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    yellow_contours, yellow_hierarchy = cv2.findContours(
        yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_yellow = cv2.drawContours(
        frame, yellow_contours, -1, (0, 255, 255), 3)
    if len(yellow_contours):
        M = cv2.moments(yellow_contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [yellow_contours[0]], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            frame, angle = getrotangle(frame, yellow_contours)
            return [True, cx, cy, output_yellow, angle]
    else:
        return [False, 0, 0, frame, 0]


def getgreencoord(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    green_lower = np.array([65, 77, 83])
    green_upper = np.array([93, 145, 121])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    segmented_green = cv2.bitwise_and(frame, frame, mask=green_mask)
    green_contours, green_hierarchy = cv2.findContours(
        green_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_green = cv2.drawContours(frame, green_contours, -1, (0, 255, 0), 3)
    if len(green_contours):
        M = cv2.moments(green_contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [green_contours[0]], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            frame, angle = getrotangle(frame, green_contours)
            return [True, cx, cy, output_green, angle]
    else:
        return [False, 0, 0, frame, 0]


def getorangecoord(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    orange_lower = np.array([6, 0, 160])
    orange_upper = np.array([14, 159, 240])
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    segmented_orange = cv2.bitwise_and(frame, frame, mask=orange_mask)
    orange_contours, orange_hierarchy = cv2.findContours(
        orange_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_orange = cv2.drawContours(
        frame, orange_contours, -1, (0, 165, 255), 3)
    if len(orange_contours):
        M = cv2.moments(orange_contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [orange_contours[0]], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            frame, angle = getrotangle(frame, orange_contours)
            return [True, cx, cy, output_orange, angle]
    else:
        return [False, 0, 0, frame, 0]


def getwhitecoord(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    white_lower = np.array([0, 0, 210])
    white_upper = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    segmented_white = cv2.bitwise_and(frame, frame, mask=white_mask)
    white_contours, white_hierarchy = cv2.findContours(
        white_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_white = cv2.drawContours(
        frame, white_contours, -1, (0, 165, 255), 3)
    if len(white_contours):
        M = cv2.moments(white_contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(
                frame, [white_contours[0]], -1, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            frame, angle = getrotangle(frame, white_contours)
            return [True, cx, cy, output_white, angle]
    else:
        return [False, 0, 0, frame, 0]


def getpurplecoord(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    purple_lower = np.array([88, 20, 0])
    purple_upper = np.array([131, 126, 169])
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
    segmented_purple = cv2.bitwise_and(frame, frame, mask=purple_mask)
    purple_contours, purple_hierarchy = cv2.findContours(
        purple_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_purple = cv2.drawContours(
        frame, purple_contours, -1, (0, 165, 255), 3)
    if len(purple_contours):
        M = cv2.moments(purple_contours[0])
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, [purple_contours[0]], -1, (272, 75, 54), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            print(f"x: {cx} y: {cy}")
            frame, angle = getrotangle(frame, purple_contours)
            return [True, cx, cy, output_purple, angle]
    else:
        return [False, 0, 0, frame, 0]


def getframe():
    _, frame = cam.read()
    return frame


def getnextframe(colors, frame):
    detect = False
    if (colors[0] == True) and (detect == False):
        detect, x, y, newframe, angle = getbluecoord(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "b", rotangle)
        else:
            pass
    if (colors[1] == True) and (detect == False):
        detect, x, y, newframe, angle = getredcoord(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "r", rotangle)
        else:
            pass
    if (colors[2] == True) and (detect == False):
        detect, x, y, newframe, angle = getyellowcoord(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "y", rotangle)
        else:
            pass
    if (colors[3] == True) and (detect == False):
        detect, x, y, newframe, angle = getgreencoord(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "g", rotangle)
        else:
            pass
    if (colors[4] == True) and (detect == False):
        detect, x, y, newframe, angle = getorangecoord(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "o", rotangle)
        else:
            pass
    if (colors[5] == True) and (detect == False):
        detect, x, y, newframe, angle = getwhitecoord(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "w", rotangle)
        else:
            pass
    if (colors[6] == True) and (detect == False):
        detect, x, y, newframe, angle = getpurplecoord(frame)
        if detect:
            rotangle = angle  # - "getal"
            return (newframe, x, y, "p", rotangle)
        else:
            pass
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
