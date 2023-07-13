import argparse
import os
import sys
import pandas as pd

import configure
import reportprocess
from configure import DEFAULT_OUTPUTFILE
from evaluate.assess import V_measure,assessall_VDP
from image.widget import ocr_boxes
from util import view_bar
from evaluate.sequences import gen_sequences, gen_all_sequences_VDP
import image.image_main as img
import text.text_main as txt

def processoneapp_norm(appnum):
    reports = reportprocess.getreports(appnum)
    st_list, ct_list, ct_global_max, ct_list_c, ct_global_max_c = img.image_main(reports)
    p_list, r_list = txt.text_main(reports)
    print(ct_list, st_list, p_list, r_list)
    sequences = gen_all_sequences_VDP(st_list, ct_list, p_list,r_list, reports)
    evaluate_result_dict, evaluate_result_forcsv = assessall_VDP(sequences)
    return evaluate_result_forcsv


def singleprocess(args):
    if args.apprange is not None:
        range_app = range(args.apprange[0], args.apprange[1] + 1)
    else:
        range_app = range(1, 21)
    if args.onlyimage:
        import image.image_main as img
        for i in range_app:
            print('start app %d' % i)
            reports = reportprocess.getreports(i)
            st_list, ct_list, ct_global_max, ct_list_c, ct_global_max_c = img.image_main(reports)
            print(st_list, ct_list, ct_global_max, ct_list_c, ct_global_max_c)
    if args.onlytext:
        import text.text_main as txt
        for i in range_app:
            print('start app %d' % i)
            reports = reportprocess.getreports(i)
            p_list, r_list = txt.text_main(reports)
            print(p_list, r_list)



def main(args):
    if args.initialall:
        reports = reportprocess.getallreports()
        for r in reports:
            ocr_boxes(configure.IMAGE_DIR + r.imagename)
            view_bar(reports.index(r), len(reports), 'ocr initial', str(r) + ' initialed')
        return
    if args.onlyimage or args.onlytext:
        singleprocess(args)
        return
    result = {}
    if args.evaluateall:
        for i in range(1, 21):
            result['app' + str(i)] = processoneapp_norm(i)
    else:
        result['app' + str(args.appnum)] = processoneapp_norm(args.appnum)
    df = pd.DataFrame(result)
    print(df)
    df.to_csv(args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--appnum', type=int, default=1, choices=list(range(1, 21)),
                        help='The reports of what app to be operated')
    parser.add_argument('--apprange', type=int, nargs='+',
                        help='The reports of what range of app to be operated')
    parser.add_argument('--evaluateall', action='store_true', default=False,
                        help='Will all the result be evaluated')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUTFILE, help='The file to save the result')
    parser.add_argument('--initialall', action='store_true', default=False, help='initial all data')
    parser.add_argument('--onlyimage', action='store_true', default=False, help='only image be processed')
    parser.add_argument('--onlytext', action='store_true', default=False, help='only text be processed')
    return parser.parse_args(argv)


if __name__ == '__main__':
    if len(sys.argv)>1:
        main(parse_arguments(sys.argv[1:]))
    else:
        main(parse_arguments('--initialall'.split(' ')))
        main(parse_arguments('--evaluateall --output ./file/result_vdp'.split(' ')))

