import os
from typing import List

import cv2
import pandas as pd
import xml.dom.minidom as xmldom

import image.vgg16 as vgg16
import util
from configure import IOU_BIAS, IOU_THRESHOLD
from mathutil import insameclass
from reportprocess import Report


def iou_match(list1, list2, bias=IOU_BIAS,threshold = IOU_THRESHOLD):  # bias: enlarge inter rect
    col_min_a, row_min_a, col_max_a, row_max_a = int(list1[1]), int(list1[0]), \
                                                 int(list1[1]) + int(list1[2]), int(list1[0]) + int(list1[3])
    col_min_b, row_min_b, col_max_b, row_max_b = int(list2[1]), int(list2[0]), \
                                                 int(list2[1]) + int(list2[2]), int(list2[0]) + int(list2[3])
    if col_min_a > col_max_b or col_min_b > col_max_a or row_min_a > row_max_b or row_min_b > row_max_a:  # if no inter
        return False
    col_min_s = max(col_min_a - bias, col_min_b - bias)
    row_min_s = max(row_min_a - bias, row_min_b - bias)
    col_max_s = min(col_max_a + bias, col_max_b + bias)
    row_max_s = min(row_max_a + bias, row_max_b + bias)
    w = max(0, col_max_s - col_min_s)
    h = max(0, row_max_s - row_min_s)
    inter = w * h
    area_a = (col_max_a - col_min_a) * (row_max_a - row_min_a)
    area_b = (col_max_b - col_min_b) * (row_max_b - row_min_b)
    iou = inter / (area_a + area_b - inter)  # the Si/Sa+Sb-Si  Big bug: str+str,sometimes divide 0
    return iou >= threshold


def is_pure(com):
    baseline = com[0, 0]
    base_b = baseline[0]
    base_g = baseline[1]
    base_r = baseline[2]
    height, width = com.shape[0], com.shape[1]
    for i in range(height):
        for j in range(width):
            cur_pixel = com[i, j]
            cur_b = cur_pixel[0]
            cur_g = cur_pixel[1]
            cur_r = cur_pixel[2]
            if cur_b != base_b or cur_g != base_g or cur_r != base_r:
                return False
    return True


def gen_CT_dis(xmlpath1, xmlpath2, imgpath1, imgpath2):
    dom_obj1 = xmldom.parse(xmlpath1)
    dom_obj2 = xmldom.parse(xmlpath2)
    element_obj1 = dom_obj1.documentElement
    element_obj2 = dom_obj2.documentElement
    sub_element_obj1 = element_obj1.getElementsByTagName("col")
    sub_element_obj2 = element_obj2.getElementsByTagName("col")
    list_file1_all = []
    list_file2_all = []

    isChange = False
    # get all col info (may change order)
    if len(sub_element_obj1) < len(sub_element_obj2):
        sub_element_obj1, sub_element_obj2 = sub_element_obj2, sub_element_obj1  # big small
        isChange = True
        for i in range(len(sub_element_obj1)):
            list_temp = [sub_element_obj1[i].getAttribute("x"),
                         sub_element_obj1[i].getAttribute("y"),
                         sub_element_obj1[i].getAttribute("w"),
                         sub_element_obj1[i].getAttribute("h")]
            list_file1_all.append(list_temp)
        for i in range(len(sub_element_obj2)):
            list_temp = [sub_element_obj2[i].getAttribute("x"),
                         sub_element_obj2[i].getAttribute("y"),
                         sub_element_obj2[i].getAttribute("w"),
                         sub_element_obj2[i].getAttribute("h")]
            list_file2_all.append(list_temp)
    else:
        for i in range(len(sub_element_obj1)):
            list_temp = [sub_element_obj1[i].getAttribute("x"),
                         sub_element_obj1[i].getAttribute("y"),
                         sub_element_obj1[i].getAttribute("w"),
                         sub_element_obj1[i].getAttribute("h")]
            list_file1_all.append(list_temp)
        for i in range(len(sub_element_obj2)):
            list_temp = [sub_element_obj2[i].getAttribute("x"),
                         sub_element_obj2[i].getAttribute("y"),
                         sub_element_obj2[i].getAttribute("w"),
                         sub_element_obj2[i].getAttribute("h")]
            list_file2_all.append(list_temp)

    count = 0
    flags = [False] * len(list_file2_all)
    # match rects
    match_pool = []
    # find all piece if position matched
    for i in range(len(list_file2_all)):
        for j in range(len(list_file1_all)):
            if flags[i] == False and iou_match(list_file1_all[j], list_file2_all[i]):  # has big inter
                list_t = []
                count += 1
                flags[i] = True
                list_t.append(list_file1_all[j])
                list_t.append(list_file2_all[i])
                match_pool.append(list_t)
                break
    list_match_com = []
    if isChange:
        img1 = cv2.imread(imgpath2)
        img2 = cv2.imread(imgpath1)
    else:
        img1 = cv2.imread(imgpath1)
        img2 = cv2.imread(imgpath2)

    reduce_count = count
    # cut them
    for i in range(count):
        list_temp = []
        list_pairs = match_pool[i]
        list_pair1 = list_pairs[0]
        list_pair2 = list_pairs[1]
        for k in range(4):
            list_pair1[k] = int(list_pair1[k])
            list_pair2[k] = int(list_pair2[k])
        if list_pair1[2] == 0 or list_pair1[3] == 0 or list_pair2[2] == 0 or list_pair2[3] == 0:
            continue
        com1 = img1[list_pair1[1]:list_pair1[1] + list_pair1[3],
               list_pair1[0]:list_pair1[0] + list_pair1[2]]
        if is_pure(com1):
            reduce_count -= 1
            continue
        com1 = cv2.resize(com1, (224, 224))
        com2 = img2[list_pair2[1]:list_pair2[1] + list_pair2[3],
               list_pair2[0]:list_pair2[0] + list_pair2[2]]
        com2 = cv2.resize(com2, (224, 224))
        list_temp.append(com1)
        list_temp.append(com2)
        list_match_com.append(list_temp)
    # get all distance

    distance_list, c_distance_list = vgg16.c_distance(list_match_com)
    result = 0
    result_c = 0
    for i in range(len(distance_list)):
        result += distance_list[i]
        result_c += c_distance_list[i]
    if reduce_count == 0:
        result = 1
        result_c = 1
    if reduce_count != 0:
        result /= reduce_count
        result_c /= reduce_count
    if imgpath1 == imgpath2:
        result = 0.0
        result_c = 0.0
    return result, result_c


def get_CT_matrix(xml_dir, img_dir, reports: List[Report], report_class=None):
    all_dist_list = [["index"] + [r.index for r in reports]] + [[r.index] + [""] * len(reports) for r in reports]
    all_dist_list_c = [["index"] + [r.index for r in reports]] + [[r.index] + [""] * len(reports) for r in reports]
    counter = 0
    l = len(reports)
    total = (l * l + l) / 2 if report_class is None else sum([len(c) * len(c) / 2 for c in report_class])
    global_content_distance_max = 0
    global_content_distance_max_c = 0
    for i, r_i in enumerate(reports):
        all_dist_list[i + 1][i + 1] = 0.0
        all_dist_list_c[i + 1][i + 1] = 0.0
        for j in range(i + 1, len(reports)):
            if report_class is not None and not insameclass(report_class, reports[i].index, reports[j].index):
                dis_cur = dis_cur_c = 'Nan'
            else:
                counter += 1
                r_j = reports[j]
                xml_cur1 = xml_dir + r_i.name + ".xml"
                xml_cur2 = xml_dir + r_j.name + ".xml"
                img_cur1 = img_dir + r_i.imagename
                img_cur2 = img_dir + r_j.imagename
                print(r_i, r_j, i, j)
                dis_cur, dis_cur_c = gen_CT_dis(xml_cur1, xml_cur2, img_cur1, img_cur2)
                util.view_bar(counter, total, 'GenerateCTDis')
                dis_cur = round(dis_cur, 4)
                dis_cur_c = round(dis_cur_c, 4)
                if dis_cur > global_content_distance_max:
                    global_content_distance_max = dis_cur
                if dis_cur_c > global_content_distance_max_c:
                    global_content_distance_max_c = dis_cur_c
            all_dist_list[i + 1][j + 1] = all_dist_list[j + 1][i + 1] = dis_cur
            all_dist_list_c[i + 1][j + 1] = all_dist_list_c[j + 1][i + 1] = dis_cur_c
    print('GenerateCTDis completed!')
    for i in range(1, len(all_dist_list)):
        for j in range(1, len(all_dist_list[0])):
            if all_dist_list[i][j] != 'Nan':
                all_dist_list[i][j] /= global_content_distance_max
                all_dist_list[i][j] = round(all_dist_list[i][j], 4)
        all_dist_list[i][0] = int(all_dist_list[i][0])
    for i in range(1, len(all_dist_list_c)):
        for j in range(1, len(all_dist_list_c[0])):
            if all_dist_list_c[i][j] != 'Nan':
                all_dist_list_c[i][j] /= global_content_distance_max_c
                all_dist_list_c[i][j] = round(all_dist_list_c[i][j], 4)
        all_dist_list_c[i][0] = int(all_dist_list_c[i][0])
    return all_dist_list, global_content_distance_max, all_dist_list_c, global_content_distance_max_c
