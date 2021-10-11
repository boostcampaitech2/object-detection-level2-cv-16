import os
import copy
import pandas as pd

def adjust_size(tmp):
    if int(tmp) == 0:
        tmp = 0
    elif int(tmp) == 1024:
        tmp = 1024
    return tmp

def main():
    DETECT_TXT_DIR = 'yolor_paper/yolor/runs/yolor-e6/exp/labels'
    IMAGE_DIR = 'yolor_paper/yolor/trash_data/images/test'
    SUBMISSION_NAME = './yolor_e6_best_submission_resize.csv'

    whole = sorted(os.listdir(IMAGE_DIR))
    label_name = [x[:-3]+'txt' for x in whole]

    file_names = []
    prediction_strings = []

    for i in range(len(whole)):
        file = DETECT_TXT_DIR + '/' + label_name[i]
        image_id = whole[i][:4] + '/' + whole[i][5:]
        if os.path.isfile(file):
            f = open(file)
            line = f.readline()
            predict = ''
            while line:
                line = map(float, line.rstrip().split())
                label, xc, yc, w, h, score  = line
                w *= 1024
                h *= 1024
                x = xc * 1024 - w / 2
                y = yc * 1024 - h / 2
                predict += str(int(label)) + ' ' + str(score) + ' ' + str(adjust_size(x)) + ' ' + str(adjust_size(y)) + ' ' + str(adjust_size(x + w)) + ' ' + str(adjust_size(y + h)) + ' '
                line = f.readline()
        else:
            predict = ''
        file_names.append(image_id)
        prediction_strings.append(predict)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(SUBMISSION_NAME, index=None)
    print(submission.head())
    
if __name__ == '__main__':
    main()
