import numpy as np


def csv2matrix(list):
    return np.array([s[1:] for s in list[1:]])


def gaussian_kernal(matrix, delta=1):
    return np.exp(- matrix ** 2 / (2. * delta ** 2))


def insameclass(report_class,index1,index2):
    for c in report_class:
        for r in c:
            if r.index==index1:
                for r2 in c:
                    if r2.index==index2:
                        return True
    return False