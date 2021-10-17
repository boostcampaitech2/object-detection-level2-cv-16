# Passion-ate🔥
| [강재현](https://github.com/AshHyun) | [김민준](https://github.com/danny0628) | [박상현](https://github.com/hyun06000) | [서광채](https://github.com/Gwang-chae) | [오하은](https://github.com/Haeun-Oh) | [이승우](https://github.com/DaleLeeCoding) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| ![image](https://user-images.githubusercontent.com/65941859/137628452-e2f573fe-0143-46b1-925d-bc58b2317474.png) | ![image](https://user-images.githubusercontent.com/65941859/137628521-10453cac-ca96-4df8-8ca0-b5b0d00930c0.png) | ![image](https://user-images.githubusercontent.com/65941859/137628500-342394c3-3bbe-4905-984b-48fae5fc75d6.png) | ![image](https://user-images.githubusercontent.com/65941859/137628535-9afd4035-8014-475c-899e-77304950c190.png) | ![image](https://user-images.githubusercontent.com/65941859/137628474-e9c4ab46-0a51-4a66-9109-7462d3a7ead1.png) | ![image](https://user-images.githubusercontent.com/65941859/137628443-c032259e-7a7a-4c2d-891a-7db09b42d27b.png) |
|  | [blog](https://danny0628.tistory.com/) |  |[Notion](https://kcseo25.notion.site/
) |  |  | [Notion](https://leeseungwoo.notion.site/) |


# Overview
![overview](https://user-images.githubusercontent.com/65941859/137625407-68686b0c-fe72-430e-8b2b-e52701bdd0f7.PNG)

# Project Structure
![인포](https://user-images.githubusercontent.com/65941859/137625355-0b2a647c-7680-4d6c-9e2e-d718989ca8ef.PNG)

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

