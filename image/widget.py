import os
import cv2
import sys
import json
import shutil
import numpy as np
from aip import AipOcr

import configure
import util


def canny_boxes(image, canny_sigma=0.33, dilate_count=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower_threshold = int(max(0, (1 - canny_sigma) * v))
    upper_threshold = int(min(255, (1 + canny_sigma) * v))
    # get Contour, then find the rect
    img_binary = cv2.Canny(gray, lower_threshold, upper_threshold, -1)
    img_dilated = cv2.dilate(img_binary, None, iterations=dilate_count)
    _, contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # _,contours, _ OR contours, _
    boundings = [cv2.boundingRect(c) for c in contours]
    return boundings


def ocr_request(image_bytes, lang='CHN_ENG', show_char=False):
    client = AipOcr(configure.APP_ID, configure.API_KEY, configure.SECRET_KEY)
    options = {
        'language-type': lang,
        'recognize_granularity': 'small' if show_char else 'big',
        'probability': 'true'
    }
    return client.general(image_bytes, options)


def ocr_boxes(image_path, lang='CHN_ENG', prob=0.90, space_ratio=configure.OCR_SPACE_RATIO):
    cache_dir = 'widget_tmp'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, '%s.json' %
                              os.path.basename(image_path)[0: os.path.basename(image_path).index(".")])
    #if run once, use cache
    if os.path.isfile(cache_file):
        result = json.load(open(cache_file, 'r', encoding='utf-8'))
        load_flag = True
    else:
        result = ocr_request(open(image_path, 'rb').read(), lang)
        load_flag = False

    image = cv2.imread(image_path)
    assert image is not None, 'Cannot read the image file %s.' % image_path
    if 'words_result' not in result:
        print('OCR failed: %s' % str(result), file=sys.stderr)
        if 'error_msg' in result.keys():
            raise Exception('Ocr failed due to %s'%str(result['error_msg']))
        return []

    text_boxes = []
    for words in result['words_result']:
        left = words['location']['left']
        top = words['location']['top']
        width = words['location']['width']
        height = words['location']['height']
        cropped = cv2.imencode('.jpg', image[top:top + height, left:left + width, :])[1].tobytes()
        if load_flag:
            details = words['details']
        else:
            #for every word, add new ocr result as 'details', and will add to json
            words['details'] = details = ocr_request(cropped, show_char=True)
        if 'words_result' not in details:
            continue
        for detailed_words in details['words_result']:
            #if prob is too small, neglect it
            if detailed_words['probability']['average'] < prob:
                continue
            split_idx = []
            #if the interval of two char is far, split them
            for i, (p, q) in enumerate(zip(detailed_words['chars'][:-1], detailed_words['chars'][1:])):
                distance = q['location']['left'] - p['location']['left'] - p['location']['width']
                threshold = space_ratio * min(p['location']['height'], q['location']['height'])
                if distance > threshold:
                    split_idx.append(i + 1)
            #for every spilit, cal their bounding
            for i, j in zip([0] + split_idx, split_idx + [len(detailed_words['chars'])]):
                texts = detailed_words['words'][i:j]
                q_loc = detailed_words['chars'][i]['location']
                box_x = q_loc['left'] + left
                box_y = q_loc['top'] + top
                r_loc = detailed_words['chars'][j - 1]['location']
                box_w = r_loc['left'] + r_loc['width'] - q_loc['left']
                box_h = q_loc['height']
                text_boxes.append((box_x, box_y, box_w, box_h))
    if not load_flag:
        json.dump(result, open(cache_file, 'w',encoding='utf-8'), ensure_ascii=False, indent=4)
    #shutil.rmtree('widget_tmp')
    #return boundings
    return text_boxes


def intersect(rect_a, rect_b):
    a_x, a_y, a_w, a_h = rect_a
    b_x, b_y, b_w, b_h = rect_b
    dx = max(0, min(a_x + a_w, b_x + b_w) - max(a_x, b_x))
    dy = max(0, min(a_y + a_h, b_y + b_h) - max(a_y, b_y))
    S_i = dx * dy
    S_a = a_w * a_h
    S_b = b_w * b_h
    return S_i / S_a, S_i / S_b, S_i / (S_a + S_b - S_i)


def get_boxes(image_path, threshold=.70):
    image = cv2.imread(image_path)
    assert image is not None, 'Cannot read the image file %s.' % image_path
    boundings = canny_boxes(image)
    ocr_res = ocr_boxes(image_path)#get rect in image used ocr
    mods = []
    for rect_o in ocr_res:
        best_match = (None, 0, 0, 0)
        for rect_c in boundings:#get the rect_c whose ratio with rect_o is max
            ratio_o, ratio_c, ratio = intersect(rect_o, rect_c)#get Si/Sa,Si/Sb,Si/(Sa+Sb-Si)
            if ratio > best_match[3]:
                best_match = (rect_c, ratio_o, ratio_c, ratio)
        rect_c, _, ratio_c, _ = best_match
        if rect_c is not None:
            if ratio_c > threshold:#if rect_o almost contain rect_c, replace rect_c
                mods.append(('replace', rect_c, rect_o))
                for rect_cc in boundings:#delete all other rect in boundings which like rect_c(contained by rect_o)
                    if rect_cc == rect_c:
                        continue
                    _, ratio_cc, _ = intersect(rect_o, rect_cc)
                    if ratio_cc > threshold:
                        mods.append(('delete', rect_cc))
            else:#else add rect_o
                mods.append(('add', rect_o))
        else:
            mods.append(('add', rect_o))
    for mod in set(mods):
        if mod[0] == 'replace':
            boundings[boundings.index(mod[1])] = mod[2]
        elif mod[0] == 'add':
            boundings.append(mod[1])
        elif mod[0] == 'delete':
            boundings.remove(mod[1])
    #util.show_boxes(image_path,boundings,'last')
    return boundings
