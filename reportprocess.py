import os
from urllib.request import urlretrieve

import pandas as pd

import configure


class Report:
    def __init__(self, index, app, category, text, imageurl):
        self.index = index
        self.app = app
        self.category = category
        self.text = text
        self.imageurl = imageurl
        self.name = f"app{app}_{category}_num{index}"
        self.imagename=self.name + '.'+imageurl.split('.')[-1]
    def __str__(self):
        return '%d_%d_%d'%(self.index,self.app,self.category)
    def __repr__(self):
        return self.__str__()
    def __hash__(self):
        return int(self.index)
    def __eq__(self, other):
        return self.index==other.index


def download_image(report):
    path = os.path.join(configure.IMAGE_DIR, report.imagename)
    if os.path.exists(path):
        return
    urlretrieve(report.imageurl, path)
    print(report,' downloaded')


def getallreports():
    reports = pd.read_csv(configure.REPORT_FILE_PATH)
    # 0 of itertuples is index
    reports = [Report(i[1], i[2], i[3], i[4], i[5]) for i in reports.itertuples()]
    for r in reports:
        download_image(r)
    return reports


def getreports(appnum):
    reports = pd.read_csv(configure.REPORT_FILE_PATH)
    # 0 of itertuples is index
    reports = [Report(i[1], i[2], i[3], i[4], i[5]) for i in reports.itertuples()]
    for r in reports:
        download_image(r)
    return [r for r in reports if r.app == appnum]
