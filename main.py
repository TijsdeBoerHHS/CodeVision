import cv2
import numpy as np
from time import sleep
from struct import pack
from tcp import send_data, receive_data
from toolbox import find_color_threshold
from vision import get_coordinate_factory

print('Connecting to the camera...')
cam = cv2.VideoCapture(0)
cam.set(10,50)
print('Connected')


colorList = {
    'blue': {
        'lower': [108, 181, 115],
        'upper': [117, 255, 220],
        'colors': {
            'contour': (255, 0, 0),
            'text': (255, 255, 255),
            'center': (0, 0, 255),
        }
    },
    'red': {
        'lower': [0, 2, 108],
        'upper': [7, 214, 255],
        'colors': {
            'contour': (0, 0, 255),
            'text': (0, 0, 0),
            'center': (255, 0, 0),
        }
    },
    'yellow': {
        'lower': [20, 185, 185],
        'upper': [34, 255, 255],
        'colors': {
            'contour': (0, 255, 255),
            'text': (0, 0, 0),
            'center': (0, 0, 255),
        }
    },
    'green': {
        'lower': [37, 61, 0],
        'upper': [68, 222, 255],
        'colors': {
            'contour': (0, 255, 0),
            'text': (255, 255, 255),
            'center': (0, 0, 255),
        }
    },
    'orange': {
        'lower': [6, 139, 126],
        'upper': [27, 253, 255],
        'colors': {
            'contour': (0, 165, 255),
            'text': (255, 255, 255),
            'center': (255, 0, 0),
        }
    },
    'white': {
        'lower': [0, 0, 168],
        'upper': [31, 41, 255],
        'colors': {
            'contour': (255, 255, 255),
            'text': (0, 0, 0),
            'center': (0, 0, 255),
        }
    },
    'purple': {
        'lower': [118, 96, 66],
        'upper': [161, 255, 255],
        'colors': {
            'contour': (174, 0, 255),
            'text': (255, 255, 255),
            'center': (0, 0, 255),
        }
    }
}


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


get_blue_coordinate = get_coordinate_factory(colorList['blue'])
get_red_coordinate = get_coordinate_factory(colorList['red'])
get_yellow_coordinate = get_coordinate_factory(colorList['yellow'])
get_green_coordinate = get_coordinate_factory(colorList['green'])
get_orange_coordinate = get_coordinate_factory(colorList['orange'])
get_white_coordinate = get_coordinate_factory(colorList['white'])
get_purple_coordinate = get_coordinate_factory(colorList['purple'])


def get_frame():
    _, frame = cam.read()
    # frame = cv2.imread('blok.jpg')
    return frame


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


def set_data_for_buffer(x, y, angle, color):
    return pack('>ffhss', x, y, angle, bytes(color[0], 'ascii'), bytes(color[1], 'ascii'))


def connection_test():
    data = set_data_for_buffer(12.2, 30.2, 30, 'blue')
    try:
        send_data(data)
        sleep(0.01)
        receive_data('>ffhss')
    except TimeoutError:
        print('Connection Timeout')
    sleep(0.1)


def main():
    find_color_threshold(get_frame, 1)
    frame = get_frame()
    resizeFactor = frame.shape[0] / 600
    resizeShape = (round(frame.shape[1] / resizeFactor), round(frame.shape[0] / resizeFactor))
    cv2.imshow("Detection", cv2.resize(frame, resizeShape))

    queueColorDetection = [
        'blue',
        'red',
        'yellow',
        'green',
        'orange',
        'white',
        'purple',
        'blue',
        'red',
        'yellow',
        'green',
        'orange',
        'white',
        'purple',
        'blue',
        'red',
        'yellow',
        'green',
        'orange',
        'white',
        'purple',
        'blue',
        'red',
        'yellow',
        'green',
        'orange',
        'white',
        'purple',
    ]
    colorsDetected = []

    # connection_test()
    test = [
        [40, 40],
        [40, -40],
        [-40, -40],
        [-40, 40],
    ]
    counter = 0
    while 1:
        counter += 1
        # connection_test
        if cv2.waitKey(0) == ord("n"):
            frame = get_frame()
            frame, x, y, msg, rotangle = get_next_blok(queueColorDetection, frame)
            #xfin,yfin = conversion(x,y)
            if msg:
                data = set_data_for_buffer(test[counter % 4][0], test[counter % 4][1], 30, 'blue')
                try:
                    send_data(data)
                except TimeoutError:
                    print('Connection Timeout')
            sleep(0.1)
            cv2.imshow("Detection", cv2.resize(frame, resizeShape))


if __name__ == "__main__":
    main()
