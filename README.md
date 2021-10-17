# Passion-ate Object Detection
# Overview
![overview](https://user-images.githubusercontent.com/65941859/137625407-68686b0c-fe72-430e-8b2b-e52701bdd0f7.PNG)

# Project Structure
![인포](https://user-images.githubusercontent.com/65941859/137625355-0b2a647c-7680-4d6c-9e2e-d718989ca8ef.PNG)

| [권태양](https://github.com/sunnight9507) | [류재희](https://github.com/JaeheeRyu) | [박종헌](https://github.com/PJHgh) | [오수지](https://github.com/ohsuz) | [이현규](https://github.com/LeeHyeonKyu) | [정익효](https://github.com/dlrgy22) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/59340911/119260030-eeae6200-bc0b-11eb-92e3-23e69ba35984.png) | ![image](https://user-images.githubusercontent.com/59340911/119260176-8f9d1d00-bc0c-11eb-9a7b-32a33c1a1072.png) | ![image](https://user-images.githubusercontent.com/59340911/119385801-a07b8a80-bd01-11eb-83c4-f3647bdd131a.png) | ![image](https://user-images.githubusercontent.com/59340911/119385429-13d0cc80-bd01-11eb-8855-8c57cdaaafc6.png) | ![image](https://user-images.githubusercontent.com/72373733/122941284-37467000-d3b0-11eb-8764-f6c7b4bed4d2.png) | ![image](https://user-images.githubusercontent.com/59340911/119260159-84e28800-bc0c-11eb-8164-6810a92bff38.png) |
| [Notion](https://www.notion.so/Sunny-1349e293c9f74de092dce9ee359bd77c) | [Notion](https://www.notion.so/AI-Research-Engineer-6f6537a7675542be901a3499e71140f9) | [Notion](https://www.notion.so/Boostcamp-deef2c0783f24c0b8022ba30b5782986) | [Notion](https://www.ohsuz.dev/) | [Notion](https://www.notion.so/thinkwisely/Naver-Boost-Camp-AI-Tech-ba743126e68749d58bdbb7af0580c8ee) |   |
| ENFJ | ISTP | INTJ | ENTP | ESTJ | INFP |
---
# Result
- private LB : 0.666 (6등)

![leaderboard](https://user-images.githubusercontent.com/65941859/137625481-fb8a113c-4d21-404d-a4c6-588dc75307fe.png)
---
# How to Use

## Installation

- `pip install -r requirements.txt`

## Model
### Swin
- Train
    - `python mmdetection/tools/train.py {CONFIG_PATH}`
- Inference
    - `python mmdetection/tools/test.py ${config} ${model} --out mmdetection/work_dirs/latest.pkl`
- Evaluation
    - `python mmdetection/make_submission.py --pkl mmdetection/work_dirs/latest.pkl --csv mmdetection/work_dirs/result.csv`
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

### Yolor
- Train
    - `python yolor/train.py --data yolor/trash_data/coco.yaml --cfg yolor/models/yolor-d6.yaml --weights 'yolor/yolor-d6.pt' --device 0 --name yolor-d6 --hyp trash_data/hyp.yaml`
- Inference
    - `python yolor/detect.py --source yolor/trash_data/images/test --weights yolor/runs/train/yolor-d6/weights/best.pt --conf 0.001 --iou 0.5 --img-size 1024 --device 0 --save-txt --save-conf --project yolor/runs/yolor-d6`
- Evaluation
    - `python yolor/yolor_make_submission.py`
```
@article{wang2021you,
  title={You Only Learn One Representation: Unified Network for Multiple Tasks},
  author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2105.04206},
  year={2021}
}
```

## PR curve Analysis
- json
    - `python tools/test.py ${CONFIG} ${CHECKPOINT_DIR} --format-only --options "jsonfile_prefix=./${FILE NAME}"`
- coco analysis
    - `python tools/analysis_tools/coco_error_analysis.py ${RESULT} ${OUT_DIR} --ann ${ANN}`
    
![image](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/55da246a-64ef-47f4-8b06-0cde944557d2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211017%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211017T110634Z&X-Amz-Expires=86400&X-Amz-Signature=2da0fdadd1c21d8fc9eec908c84798585087929e562df49424b18cf2a1434c60&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

