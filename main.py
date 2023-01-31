import cv2
import numpy as np
import toolbox as tb
import snap7
from snap7 import util

cam = cv2.VideoCapture(0)

colorList = [
    'blue',
    'red',
    'yellow',
    'green',
    'orange',
    'white',
    'purple'
]

color_ul = {
    'blue':{
        'lower':[106, 161, 98],
        'upper':[109, 255, 255]
    },
    'red':{
        'lower':[4, 53, 24],
        'upper':[8, 247, 247]
    },
    'yellow':{
        'lower':[24, 71, 105],
        'upper':[33, 255, 255]
    },
    'green':{
        'lower':[68, 79, 89],
        'upper':[100, 168, 242]
    },
    'orange':{
        'lower':[9, 162, 106],
        'upper':[20, 255, 255]
    },
    'white':{
        'lower':[0, 0, 199],
        'upper':[179, 26, 255]
    },
    'purple':{
        'lower':[122, 111, 83],
        'upper':[168, 203, 207]
    },
}


def plc_get_connection():
    client = snap7.client.Client()
    client.connect('192.168.0.15',0,0)
    client.get_connected()
    db = client.db_read(5,0,18)
    print(db)


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
    xfin = 0.16475*xrot-155.21
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


def contour_fiter(color_contours):
    color_contours = list(color_contours)
    dellist = []
    for c in range(len(color_contours)):
        area = cv2.contourArea(color_contours[c])
        if area < 1000:
            dellist.append(c)
        else:
            continue
    for i in reversed(range(len(dellist))):
        del color_contours[dellist[i]]
    return color_contours


def get_color_contours(frame, color_lower, color_upper, color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((7, 7), np.uint8)
    color_mask = cv2.inRange(hsv, color_lower, color_upper)
    cv2.imwrite('debug/color_mask1.png', color_mask)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('debug/color_mask2.png', color_mask)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('debug/color_mask3.png', color_mask)
    color_contours, _ = cv2.findContours(
        color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_contours = contour_fiter(color_contours)
    output_color = cv2.drawContours(frame, color_contours, -1, color, 3)
    return color_contours, output_color


def get_contour_coordinate(color_contours):
    if len(color_contours):
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
    color_lower = np.array(color_ul['blue']['lower'])
    color_upper = np.array(color_ul['blue']['upper'])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (255, 0, 0))

    return get_color_coordinate(frame, color_contours, output_color)


def get_red_coordinate(frame):
    color_lower = np.array(color_ul['red']['lower'])
    color_upper = np.array(color_ul['red']['upper'])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 0, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_yellow_coordinate(frame):
    color_lower = np.array(color_ul['yellow']['lower'])
    color_upper = np.array(color_ul['yellow']['upper'])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 255, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_green_coordinate(frame):
    red_lower = np.array(color_ul['green']['lower'])
    red_upper = np.array(color_ul['green']['upper'])

    color_contours, output_color = get_color_contours(frame, red_lower, red_upper, (0, 255, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_orange_coordinate(frame):
    color_lower = np.array(color_ul['orange']['lower'])
    color_upper = np.array(color_ul['orange']['upper'])
    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 165, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_white_coordinate(frame):
    color_lower = np.array(color_ul['white']['lower'])
    color_upper = np.array(color_ul['white']['upper'])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 165, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_purple_coordinate(frame):
    color_lower = np.array(color_ul['purple']['lower'])
    color_upper = np.array(color_ul['purple']['upper'])

    color_contours, output_color = get_color_contours(frame, color_lower, color_upper, (0, 165, 255))

    return get_color_coordinate(frame, color_contours, output_color)


def get_frame():
    # _, frame = cam.read()
    frame = cv2.imread('bakje3.png')
    blur = cv2.GaussianBlur(frame,(15,15),cv2.BORDER_DEFAULT)
    return blur


def get_next_blok(colors: list[str], frame):
    if not len(colors):
        print('Queue is Empty')
        return (frame, 0, 0, "QE", 0)

    detect = False
    color = colors.pop()
    print(f'Looking for {color}')

    if color not in colorList:
        print(f'color {color} is not declare in you colorList')
        return (frame, 0, 0, "QE", 0)
    detect, x, y, new_frame, angle = eval(f'get_{color}_coordinate')(frame)
    if detect:
        rotangle = angle  # - "getal"
        return (new_frame, x, y, color[0], rotangle)

    colors.append(color)
    print("Piece Error")
    # PE stands for Piece Error, which means no needed pieces found
    return (frame, 0, 0, "PE", 0)


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
    frame = get_frame()
    resizeFactor = frame.shape[0] / 600
    resizeShape = (round(frame.shape[1] / resizeFactor), round(frame.shape[0] / resizeFactor))
    cv2.imshow("Detection", cv2.resize(frame, resizeShape))

    queueColorDetection = [
        colorList[0],
        colorList[1],
        colorList[2],
        colorList[3],
        colorList[4],
        colorList[5],
        colorList[6],
    ]
    colorsDetected = [
        
    ]
    # tb.find_color_threshold(get_frame())

    while (True):
        if cv2.waitKey(0) == ord("n"):
            frame = get_frame()
            frame, x, y, msg, rotangle = get_next_blok(queueColorDetection, frame)
            # colors = checkmsg(msg, colors)
            # xfin,yfin = conversion(x,y)
            cv2.imshow("Detection", cv2.resize(frame, resizeShape))


if __name__ == "__main__":
    main()
