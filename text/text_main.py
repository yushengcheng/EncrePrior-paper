import sys
import numpy as np

import text.text_feature as rfe
from util import printnowtime


def del_invalid(p_list, r_list):
    for i in range(1, len(p_list)):
        del p_list[i][1]
    for i in range(1, len(r_list)):
        del r_list[i][1]
    del (p_list[1])
    del (r_list[1])
    return p_list, r_list


def text_main(reports):

    np.set_printoptions(threshold=sys.maxsize)

    # report feature extraction & dis matrix calculation
    text_feature_list = rfe.extract_report_feature(reports)
    p_list, r_list = rfe.cal_sim_matrix(reports, text_feature_list)
    p_list, r_list = del_invalid(p_list, r_list)
    printnowtime()

    return p_list, r_list
