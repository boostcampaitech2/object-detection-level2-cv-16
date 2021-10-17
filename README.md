# Passion-ate Object Detection

# 프로젝트 개요
- 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.
- 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.
- 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋은 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진입니다.
- 저희 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 

![image](https://user-images.githubusercontent.com/32856219/137434185-0b2417af-7b31-4ce0-b0b5-4525c0fd8356.png)


---
# Result
- private LB : 0.666 (6등)
- public LB : 0.680 (7등)
---
# How to Use

```
├── mmddetection
├── README.md
├── yolor
```
## 필수 설치

- `pip install -r requirements.txt`

## Model
### Swin
- Train
    - `python mmdetection/tools/train.py {CONFIG_PATH}`
- Inference
    - `python mmdetection/tools/test.py ${config} ${model} --out mmdetection/work_dirs/latest.pkl`
- Evaluation
    - `python mmdetection/make_submission.py --pkl mmdetection/work_dirs/latest.pkl --csv mmdetection/work_dirs/result.csv`
### Yolor
- Train
    - `python yolor/train.py --data yolor/trash_data/coco.yaml --cfg yolor/models/yolor-d6.yaml --weights 'yolor/yolor-d6.pt' --device 0 --name yolor-d6 --hyp trash_data/hyp.yaml`
- Inference
    - `python yolor/detect.py --source yolor/trash_data/images/test --weights yolor/runs/train/yolor-d6/weights/best.pt --conf 0.001 --iou 0.5 --img-size 1024 --device 0 --save-txt --save-conf --project yolor/runs/yolor-d6`
- Evaluation
    - `python yolor/yolor_make_submission.py`
##Error Analysis
- json
    - `python tools/test.py ${CONFIG} ${CHECKPOINT_DIR} --format-only --options "jsonfile_prefix=./${FILE NAME}"`
- coco analysis
    - `python tools/analysis_tools/coco_error_analysis.py ${RESULT} ${OUT_DIR} --ann ${ANN}`
![image](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/55da246a-64ef-47f4-8b06-0cde944557d2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211017%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211017T110634Z&X-Amz-Expires=86400&X-Amz-Signature=2da0fdadd1c21d8fc9eec908c84798585087929e562df49424b18cf2a1434c60&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

@article{wang2021you,
  title={You Only Learn One Representation: Unified Network for Multiple Tasks},
  author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2105.04206},
  year={2021}
}
