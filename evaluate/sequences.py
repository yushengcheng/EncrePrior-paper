import random
from copy import deepcopy

import numpy as np


def distance(q, report, matrixs, alpha=0.5, beta=0.5):
    assert alpha + beta == 1, 'alpha plus beta should 1'
    assert 0 < len(matrixs) < 3
    m_a = matrixs[0]
    q_index = m_a[0].index(q.index)
    r_index = m_a[0].index(report.index)
    dis_list = []
    for m in matrixs:
        dis_list.append(m[q_index][r_index])
    return dis_list[0] if len(matrixs) == 1 else alpha * dis_list[0] + beta * dis_list[1]


def min_prior(matrixs, reports):
    print('next seq')
    pool = []
    query = deepcopy(reports)
    pool.append(query[0])
    query.remove(query[0])
    # r = random.randint(0, len(query) - 1)
    # pool.append(query[r])
    # query.remove(query[r])
    target = None
    while len(query) != 0:
        dis1 = -1
        dis2 = -1
        for q in query:
            for report in pool:
                dis = distance(q, report, matrixs)
                if dis > dis2:
                    dis2 = dis
                else:
                    continue
            if dis2 > dis1:
                target = q
                dis1 = dis2
            else:
                continue
        pool.append(target)
        query.remove(target)
        # print('{}/{}'.format(len(pool), len(query)))
    return pool


def gen_sequences(st_list, ct_list, p_list, r_list, reports,sim=False):
    st_sequence = min_prior([st_list], reports)
    ct_sequence = min_prior([ct_list], reports)
    p_sequence = min_prior([p_list], reports)
    r_sequence = min_prior([r_list], reports)
    all_sequence = min_prior([st_list,ct_list,p_list,r_list], reports)
    all_avg_sequence, all_dict_sequence = gen_avg_dict([st_sequence, ct_sequence,p_sequence,r_sequence])
    return {'st': st_sequence, 'ct': ct_sequence, 'p': p_sequence, 'r': r_sequence,
            'all': all_sequence, 'avg': all_avg_sequence, 'dict': all_dict_sequence,
            }




def gen_avg_dict(sequences):
    report_contain = {}
    report_average = {}
    report_dict = {}

    # while 'NAN' in sequences:
    #     sequences.remove('NAN')
    # while np.nan in sequences:
    #     sequences.remove(np.nan)

    # initial dicts above
    for s in sequences:
        for i in s:
            report_average[i] = 0
            report_dict[i] = []
            if i in report_contain.keys():
                report_contain[i] += 1
            else:
                report_contain[i] = 1
    # if a report not appears in all sequences, pop it
    for i in report_contain:
        if report_contain[i] != len(sequences):
            report_average.pop(i)
            report_dict.pop(i)

    n = len(sequences)
    for s in sequences:
        for i in range(len(s)):
            try:
                report_average[s[i]] += int(i)
                report_dict[s[i]].append(int(i))
            except Exception:
                continue
    for i in report_average:
        report_average[i] /= n
        report_dict[i].sort()
    average_sequence = [i[0] for i in sorted(report_average.items(), key=lambda x: x[1])]
    dict_sequence = [i[0] for i in sorted(report_dict.items(), key=lambda x: x[1])]

    return average_sequence, dict_sequence

def gen_all_sequences_VDP( st_list, ct_list, p_list,r_list, reports):
    sequences_dict = gen_sequences(st_list, ct_list, p_list,r_list, reports)
    return sequences_dict

