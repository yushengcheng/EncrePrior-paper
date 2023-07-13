import jieba
import gensim
import numpy as np
import pandas as pd

from text.text_feature_extraction import dtw as dtw
import text.text_feature_extraction.text_feature_extraction as tfe

# pre-trained word2ve model
from util import view_bar

word2vec_model = gensim.models.Word2Vec.load('text/text_feature_extraction/bugdata_format_model_100')
# initial widget categories vector of blank report
initial_widget_categories = np.zeros(15)
# initial reproduction steps list of blank report
initial_reproduction_procedures = ['']
# initial bug descriptions list of blank report
initial_bug_descriptions = ['']


# sim between two short sentences (measured by euclidean distance)
def sentence_sim(model, s1, s2):
    size = model.layer1_size

    def sentence_vector(s):
        words = []
        try:
            words = [x for x in jieba.cut(s, cut_all=True) if x != '']
        except:
            return np.zeros(size)
        v = np.zeros(size)
        length = len(words)
        for word in words:
            try:
                v += model.wv[word]
            except:
                length -= 1
        if length == 0:
            return np.zeros(size)
        v /= length
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return eucli_distance(v1, v2)


def eucli_distance(a, b):
    dis = np.sqrt(sum(np.power((a - b), 2)))
    return round(dis, 4)


# normalize distance to [0,1] by max_dis and min_dis
def normalize_dis(dis, max_dis, min_dis):
    return (dis - min_dis) / (max_dis - min_dis)


# SR = gamma * SB + delta * SC
def cal_report_similarity(text_feature1, text_feature2, index1, index2, max_problem_dis, min_problem_dis,
                          max_procedure_dis, min_procedure_dis):
    sp_dis = cal_bug_similarity(text_feature1, text_feature2, max_problem_dis, min_problem_dis)
    # if SB == -1 or SB == -2:
    #     return SB
    sr_dis = cal_context_similarity(text_feature1, text_feature2, max_procedure_dis, min_procedure_dis)
    # SR = gamma * SB + delta * SC
    # whole_dis = 0.25 * sp_dis + 0.25 * swp_dis + 0.25 * sr_dis + 0.25 * swc_dis
    return round(sp_dis, 4), round(sr_dis, 4)


# SB = alpha1 * SP + beta1 * SWP
def cal_bug_similarity(text_feature1, text_feature2, max_problem_dis, min_problem_dis):
    problem_list1 = text_feature1['problems_list']
    problem1 = ' '.join(problem_list1)
    problem2 = ''
    if text_feature2 is not None:
        problem_list2 = text_feature2['problems_list']
        problem2 = ' '.join(problem_list2)
    # dis = sift_similarity(pwidget_img1, pwidget_img2)
    # SWP = normalize_dis(dis, max_sift_fea, 0)
    # swp_dis = 1 - SWP
    # if SWP == -1 or SWP == -2:
    #     return SWP
    dis = sentence_sim(word2vec_model, problem1, problem2)
    sp_dis = sentence_sim(word2vec_model, problem1, problem2)
    sp_dis = normalize_dis(sp_dis, max_problem_dis, min_problem_dis)
    # SB = alpha1 * SP + beta1 * SWP
    # print('SP:{}'.format(SP))
    return sp_dis


# SC = alpha2 * SR + beta2 * SWC
def cal_context_similarity(text_feature1, text_feature2, max_procedure_dis, min_procedure_dis):
    procedures_list1 = text_feature1['procedures_list']
    procedures_list2 = initial_reproduction_procedures
    if text_feature2 is not None:
        procedures_list2 = text_feature2['procedures_list']
    # sim = 1 - dis
    sr_dis = dtw.dtw_distance(procedures_list1, procedures_list2, min_procedure_dis, max_procedure_dis)
    # SR = 1.0 - sr_dis
    # cal dis between two widget category vec
    # print('$$$$$$$$$$$$$$$$$$$$$')
    # print(category_vec1)
    # print(category_vec2)
    # print('$$$$$$$$$$$$$$$$$$$$$')
    # dis = normalize_dis(eucli_distance(category_vec1, category_vec2), max_category_dis, min_category_dis)
    # swc_dis = round(dis, 4)
    # SWC = 1.0 - swc_dis
    # SC = alpha2 * SR + beta2 * SWC
    sr_dis = normalize_dis(sr_dis, max_procedure_dis, min_procedure_dis)
    # print('SR:{}'.format(SR))
    return sr_dis


# get max and min value of four types of different distance
def cal_normalize_dis(number, text_feature_list):
    # the max num of sift feature points of screenshots (used to normalize dis)
    max_sift_fea = 0.
    # the max val of Euclidean distance of two bug descriptions (used to normalize dis)
    max_problem_dis = 0.
    # the min val of Euclidean distance of two bug descriptions (used to normalize dis)
    min_problem_dis = 100000.
    # the max val of Euclidean distance of two reproduction steps (used to normalize dis)
    max_procedure_dis = 0.
    # the min val of Euclidean distance of two reproduction steps (used to normalize dis)
    min_procedure_dis = 100000.
    # the max val of Euclidean distance of two widget categories vectors (used to normalize dis)
    max_category_dis = 0.
    # the min val of Euclidean distance of two widget categories vectors (used to normalize dis)
    min_category_dis = 100000.
    for i in range(0, len(number)):
        # cal dis between each report and blank report
        #print('-------------start cal sim between {}th & -1 report-----------'.format(number[i]))
        # fea_num = get_fea_num(problem_widget_path_list[i])
        # if fea_num > max_sift_fea:
        #     max_sift_fea = fea_num
        text_feature = text_feature_list[i]
        problem_list = text_feature['problems_list']
        #print('-----------{}th report problem_list:{}'.format(number[i], problem_list))
        problem = ' '.join(problem_list)
        problem_sim = sentence_sim(word2vec_model, problem, '')
        if problem_sim > max_problem_dis:
            max_problem_dis = problem_sim
        if problem_sim < min_problem_dis:
            min_problem_dis = problem_sim
        procedures_list = text_feature['procedures_list']
        #print('-----------{}th report procedure_list:{}'.format(number[i], procedures_list))
        if len(procedures_list)==0:
            procedures_list.append('')
        min_dtw_dis, max_dtw_dis = dtw.dtw_distance(procedures_list, initial_reproduction_procedures, 0, 0)
        if min_dtw_dis < min_procedure_dis:
            min_procedure_dis = min_dtw_dis
        if max_dtw_dis > max_procedure_dis:
            max_procedure_dis = max_dtw_dis
        # eucli_dis = eucli_distance(widget_category_list[i], initial_widget_categories)
        # if eucli_dis > max_category_dis:
        #     max_category_dis = eucli_dis
        # if eucli_dis < min_category_dis:
        #     min_category_dis = eucli_dis
        # print('-------------end cal sim between {}th & -1 report-----------'.format(number[i]))
    # cal dis between every two reports in report list
    counter=0
    for i in range(0, len(number) - 1):
        for j in range(i + 1, len(number)):
            counter+=1
            # print('-------------start cal sim between {}th & {}th report-----------'.format(number[i], number[j]))
            text_feature1 = text_feature_list[i]
            problem_list1 = text_feature1['problems_list']
            problem1 = ' '.join(problem_list1)
            text_feature2 = text_feature_list[j]
            problem_list2 = text_feature2['problems_list']
            # print('-----------{}th report problem_list:{}'.format(number[i], problem_list1))
            # print('-----------{}th report problem_list:{}'.format(number[j], problem_list2))
            problem2 = ' '.join(problem_list2)
            problem_sim = sentence_sim(word2vec_model, problem1, problem2)
            if problem_sim > max_problem_dis:
                max_problem_dis = problem_sim
            if problem_sim < min_problem_dis:
                min_problem_dis = problem_sim
            procedures_list1 = text_feature1['procedures_list']
            procedures_list2 = text_feature2['procedures_list']
            # print('-----------{}th report procedure_list:{}'.format(number[i], procedures_list1))
            # print('-----------{}th report procedure_list:{}'.format(number[i], procedures_list2))
            min_dtw_dis, max_dtw_dis = dtw.dtw_distance(procedures_list1, procedures_list2, 0, 0)
            if min_dtw_dis < min_procedure_dis:
                min_procedure_dis = min_dtw_dis
            if max_dtw_dis > max_procedure_dis:
                max_procedure_dis = max_dtw_dis
            # eucli_dis = eucli_distance(widget_category_list[i], widget_category_list[j])
            # if eucli_dis > max_category_dis:
            #     max_category_dis = eucli_dis
            # if eucli_dis < min_category_dis:
            #     min_category_dis = eucli_dis
            view_bar(counter,len(number)*len(number)/2,'calculate text dis')
    #         print('-------------end cal sim between {}th & {}th report-----------'.format(number[i], number[j]))
    # print('max_problem={},min_problem={},max_procedure={},min_procedure={}'.format(max_problem_dis, min_problem_dis,
    #                                                                                max_procedure_dis,
    #                                                                                min_procedure_dis))
    return max_problem_dis, min_problem_dis, max_procedure_dis, min_procedure_dis


# calculate and record sim between every two reports in report list
def cal_sim_matrix(reports, text_feature_list):
    number = [r.index for r in reports]
    # get max and min value of distance (used for distance normalization)
    max_problem_dis, min_problem_dis, max_procedure_dis, min_procedure_dis = \
        cal_normalize_dis(number, text_feature_list)

    header = ['index']
    for n in number:
        header.append(str(n))

    all_dis_p_list = []
    all_dis_r_list = []
    all_dis_p_list.append(header)
    all_dis_r_list.append(header)
    sp_dis_matrix = np.zeros((len(number) + 1, len(number) + 1))
    sr_dis_matrix = np.zeros((len(number) + 1, len(number) + 1))
    sp_dis_matrix[0][0] = 0
    sr_dis_matrix[0][0] = 0
    for i in range(0, len(number)):
        sp_dis, sr_dis = cal_report_similarity(text_feature_list[i], None, number[i], -1, max_problem_dis,
                                               min_problem_dis, max_procedure_dis, min_procedure_dis)
        sp_dis_matrix[i + 1][0] = sp_dis
        sp_dis_matrix[0][i + 1] = sp_dis
        sr_dis_matrix[i + 1][0] = sr_dis
        sr_dis_matrix[0][i + 1] = sr_dis
    counter=0
    for i in range(0, len(number)):
        for j in range(i, len(number)):
            counter += 1
            if i == j:
                sp_dis_matrix[i + 1][j + 1] = 0
                sr_dis_matrix[i + 1][j + 1] = 0
            else:
                sr_dis, sp_dis = cal_report_similarity(text_feature_list[i], text_feature_list[j], number[i],
                                                       number[j], max_problem_dis, min_problem_dis,
                                                       max_procedure_dis, min_procedure_dis)
                sp_dis_matrix[i + 1][j + 1] = sp_dis
                sp_dis_matrix[j + 1][i + 1] = sp_dis
                sr_dis_matrix[i + 1][j + 1] = sr_dis
                sr_dis_matrix[j + 1][i + 1] = sr_dis
                view_bar(counter,(len(number) * len(number) + len(number)) / 2,'text process')
    for i in range(0, len(number) + 1):
        index = 0
        if i == 0:
            index = -1
        else:
            index = number[i - 1]
        sp_dis = [index]
        sr_dis = [index]
        for j in range(0, len(number) + 1):
            sp_dis.append(sp_dis_matrix[i][j])
            sr_dis.append(sr_dis_matrix[i][j])
        all_dis_p_list.append(sp_dis)
        all_dis_r_list.append(sr_dis)
    return all_dis_p_list, all_dis_r_list


# the main process of extract report feature
def extract_report_feature(reports):
    return tfe.text_feature_extraction([t.text for t in reports])
