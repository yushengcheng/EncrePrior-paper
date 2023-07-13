import time

import image.structure_feature
import image.content_feature
import configure
from util import printnowtime


def image_main(reports,report_class=None):

    st_list = image.structure_feature.get_ST_dis(reports, report_class)
    printnowtime()
    ct_list,global_max,ct_list_c,global_max_c = image.content_feature.get_CT_matrix(configure.XML_DIR, configure.IMAGE_DIR, reports, report_class)
    printnowtime()
    return st_list, ct_list,global_max,ct_list_c,global_max_c

