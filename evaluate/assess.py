from copy import deepcopy
from typing import List
from sklearn.metrics import homogeneity_completeness_v_measure

from reportprocess import Report
lineartype = [0.25, 0.5, 0.75, 1]

def assessall_VDP(sequences):
    dict_result = deepcopy(sequences)
    csv_result = {}
    for key in sequences:
        apfd = APFD(sequences[key])
        apfdlinear = APFDLinear(sequences[key], lineartype)
        dict_result[key] = {'apfdl_' + str(lineartype[i]): l for i, l in enumerate(apfdlinear)}
        dict_result[key].update({'sequence': dict_result[key], 'apfd': apfd})
        csv_result[key + '_apfd'] = apfd
        for i, l in enumerate(apfdlinear):
            csv_result[key + '_apfdl_' + str(lineartype[i])] = l
    return dict_result, csv_result






def APFD(reports: List[Report]):
    order = [r.index for r in reports]
    reports_dict = {}
    for i in reports:
        reports_dict[i.index] = i.category
    F = []
    for report in reports:
        if report.category not in F:
            F.append(report.category)
    F_count = len(F)
    # index of the report in order first dectect bug F[i]
    TF = []
    for f in F:
        bugid = f
        for o_index in range(len(order)):
            if reports_dict[order[o_index]] == bugid:
                TF.append(o_index)
                break
    m = len(F)
    n = len(order)
    return 1 - (sum(TF)) / (n * m) + 1 / (2 * n)


def APFDLinear(reports: List[Report], type=None):
    reports_infolist = [[r.index, r.category] for r in reports]
    order = [r.index for r in reports]
    temp = [i[0] for i in reports_infolist]
    temp2 = []
    # to delete some id in order but not in reports
    for o in order:
        if o not in temp:
            temp2.append(o)
    for i in temp2:
        order.remove(i)
    if type is None:
        type = [0.25, 0.5, 0.75, 1]
    result = []
    for i in type:
        result.append(linear(i, reports_infolist, order))
    return result


def linear(coefficient, reports_infolist, order):
    reports_dict = {}
    for i in reports_infolist:
        reports_dict[i[0]] = i[1:]
    F = []
    for report in reports_infolist:
        if (report[1]) not in F:
            F.append((report[1]))
    F_count = len(F)
    # index of the report in order first dectect bug F[i]
    TF = []
    for f in F:
        bugid = f
        for o_index in range(len(order)):
            if reports_dict[order[o_index]][0] == bugid:
                TF.append(o_index)
                break
    F_todectect = F_count * coefficient
    intq = int(F_todectect)
    frac = F_todectect - intq
    fcount = 0
    i = 0
    temp = []
    while fcount < intq:
        if reports_dict[order[i]][0] not in temp:
            temp.append(reports_dict[order[i]][0])
            fcount += 1
        i += 1
    # as i is the index of next report to solve now, it's the count of I
    I = i
    # to find a new bug need j next reports
    j = 0
    while frac != 0:
        j += 1
        if reports_dict[order[i]][0] not in temp:
            temp.append(reports_dict[order[i]][0])
            break
        i += 1

    return I + frac * j


# returns h,c,v
def V_measure(report_class):
    labels_cluster = []
    labels_true = []
    for cluster_result, c in enumerate(report_class):
        for r in c:
            labels_cluster.append(cluster_result)
            labels_true.append(r.category)
    return homogeneity_completeness_v_measure(labels_true, labels_cluster)
