import os
import copy
import pandas as pd

DETECT_TXT_DIR = 'yolor/inference/yolor_p67_best'
IMAGE_DIR = 'yolor/trash_data/images/test'
SUBMISSION_NAME = './yolor_p67_best_submission_2.csv'

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
            label, score, xc, yc, w, h  = line
            w *= 1024
            h *= 1024
            x = xc * 1024 - w / 2
            y = yc * 1024 - h / 2
            predict += str(int(label)) + ' ' + str(score) + ' ' + str(x) + ' ' + str(
                    y) + ' ' + str(x + w) + ' ' + str(y + h) + ' '
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
