import pickle
import sys
import time
import traceback

import cv2
import numpy as np

from configure import DATA_DIR


def show_boxes(img_path, boxes, title='reced'):
    img = cv2.imread(img_path)
    for b in boxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 0, 255), 2)
    show_img(img,title)



def show_img(img, title='reced'):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)


def view_bar(num, total, title='',message=''):
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    r = '\r%s [%s%s] process %s complete: %d%%, cnt: %d' % (message,"=" * rate_num, " " * (100 - rate_num), title, rate_num, num)
    sys.stdout.write(r)
    sys.stdout.flush()
def printnowtime():
    print(time.strftime('%Y-%M-%d %H:%M', time.localtime(time.time())))


