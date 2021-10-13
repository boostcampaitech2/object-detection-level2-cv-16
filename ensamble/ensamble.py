# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import time

import json

import wandb
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from easydict import EasyDict as edict
import plotly.express as px
from collections import defaultdict


# %%
test_path = "../dataset/test.json"
with open(test_path, "r") as f:
    test_json = json.load(f)
print(test_json.keys())
te_dict = edict(test_json)


# %%
df = pd.read_csv(f"./test_results/55_SwinL.csv")
df


# %%
palette = [
    'Chartreuse','BlueViolet','Crimson','DarkOrange','DodgerBlue',
    'Gold','HotPink','SkyBlue','Wheat','Tan',
]
classes=[
    'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]

# %%
def get_IoU(ref_content, comp_content):
    ( # parse content
        refr_category_id, refr_confidence,
        refr_x0, refr_y0, refr_x1, refr_y1
    ) = ref_content

    ( # parse content
        comp_category_id, comp_confidence,
        comp_x0, comp_y0, comp_x1, comp_y1
    ) = comp_content
    assert comp_category_id == refr_category_id

    # when there is no overlap
    if (comp_x1 < refr_x0 < refr_x1) and (refr_x1 < comp_x0 < comp_x1):
        return 0
    elif (comp_y1 < refr_y0 < refr_y1) and (refr_y1 < comp_y0 < comp_y1):
        return 0
    elif (refr_x0 < refr_x1 < comp_x0) and (comp_x0 < comp_x1 < refr_x0):
        return 0
    elif (refr_y0 < refr_y1 < comp_y0) and (comp_y0 < comp_y1 < refr_y0):
        return 0
    
    cross_x0,cross_y0 = max(refr_x0,comp_x0),max(refr_y0,comp_y0)
    cross_x1,cross_y1 = min(refr_x1,comp_x1),min(refr_y1,comp_y1)
    cross_box_area = (cross_x1 - cross_x0) * (cross_y1 - cross_y0)

    refr_box_area = (refr_x1 - refr_x0) * (refr_y1 - refr_y0)
    comp_box_area = (comp_x1 - comp_x0) * (comp_y1 - comp_y0)
    union_box_area = refr_box_area + comp_box_area - cross_box_area
    
    IoU = cross_box_area / union_box_area
    return IoU

def overlap_merge(yolo_content, swin_content):
    ( # parse content
        yolo_category_id, yolo_confidence,
        yolo_x0, yolo_y0, yolo_x1, yolo_y1
    ) = yolo_content

    ( # parse content
        swin_category_id, swin_confidence,
        swin_x0,swin_y0, swin_x1,swin_y1
    ) = swin_content

    assert yolo_category_id == swin_category_id
    
    # when yolo box includes swin box
    if ( (yolo_x0 <= swin_x0) and (yolo_y0 <= swin_y0) ) and \
       ( (yolo_x1 >= swin_x1) and (yolo_y1 >= swin_y1) ):
        
        if yolo_confidence >= swin_confidence:
            return yolo_content
        else : 
            return swin_content
    
    # when swin box includes yolo box
    elif ( (yolo_x0 > swin_x0) and (yolo_y0 > swin_y0) ) and \
         ( (yolo_x1 < swin_x1) and (yolo_y1 < swin_y1) ):
        
        if yolo_confidence > swin_confidence:
            return yolo_content
        else : 
            return swin_content
    

    # when they don't cover each other
    # or when Swin predicts what yolo doesn't
    else:
        return False
    


# def bbox_prune(yolo_content, swin_content, threshold):
#     merged = overlap_merge(yolo_content, swin_content)
#     if merged:
#         return merged
#     else:
#         IoU = get_IoU(yolo_content, swin_content)
        
#         # when they indicate the same instance
#         if IoU > threshold:
#             if yolo_confidence > swin_confidence:
#                 return yolo_content
#             else : 
#                 return swin_content

#         # when they indicate different instances
#         else:
#             return yolo_content, swin_content
        
    

def nms_overlap_merge(boxes, threshold):
    nms_result = []
    for refr_content in  boxes:
        instance = [refr_content]
        for comp_content in boxes:
            if refr_content == comp_content:
                continue
            else:
                IoU = get_IoU(refr_content, comp_content)
                if IoU > threshold:
                    instance.append(comp_content)
        
        
        instance.sort(
            key=lambda x : x[1], reverse=True
        )
        best_confidence = instance[0]

        if best_confidence not in nms_result:
            nms_result.append(best_confidence)
    
    merge_result = []
    for refr_content in  nms_result:
        merge_boxes = [refr_content]
        for comp_content in nms_result:
            if refr_content == comp_content:
                continue
            else:
                merged = overlap_merge(
                    refr_content, comp_content
                )
                if merged:
                    merge_boxes.append(merged)
        merge_boxes.sort(
            key=lambda x : x[1], reverse=True
        )
        best_confidence = merge_boxes[0]
        if best_confidence not in merge_result:
            merge_result.append(best_confidence)

    return merge_result

# %%
def get_all_cls_ensamble_bbox(im, min_confidence, threshold):
    ensamble_bbox = []
    for id_idx in range(10):
        
        bboxes_per_cate = dict(
            Swin=[],
            yolo=[],
        )
        
        for d_name in ["55_SwinL.csv","64_yolo.csv"]:
            model_name = d_name[3:7]
            
            df = pd.read_csv(f"./test_results/{d_name}")
            splited_anno = df[df["image_id"]==im.file_name]["PredictionString"].to_list()[0].split(" ")
            for i in range(len(splited_anno)//6):
                bbox = splited_anno[i*6:(i+1)*6]
                
                category_id, confidence  = int(bbox[0]), float(bbox[1])

                if category_id == id_idx  and confidence > min_confidence:
                    
                    # box_point extract
                    x0, y0, x1, y1 = list(map(float, bbox[2:6]))
                    
                    bboxes_per_cate[model_name].extend(
                        (category_id, confidence, x0, y0, x1, y1)
                    )

        swin_bbox = bboxes_per_cate['Swin']
        yolo_bbox = bboxes_per_cate['yolo']
        
        candi = []
        if len(yolo_bbox) > 0:
            for yolo_i in range(len(yolo_bbox)//6):
                yolo_content = yolo_bbox[yolo_i*6:(yolo_i+1)*6]
                
                if len(swin_bbox) > 0:
                    for swin_i in range(len(swin_bbox)//6):
                        swin_content = swin_bbox[swin_i*6:(swin_i+1)*6]
                    
                        merged = overlap_merge(
                            yolo_content, swin_content
                        )
                        if merged:
                            candi.append(merged)
                        else :
                            candi.append(yolo_content)
                            candi.append(swin_content)
                else:
                    candi.append(yolo_content)
        else :
            for swin_i in range(len(swin_bbox)//6):
                swin_content = swin_bbox[swin_i*6:(swin_i+1)*6]
                candi.append(swin_content)
        
        if len(candi) > 0:
            candi.sort(key=lambda x : x[1], reverse=True)
            nms_result = nms_overlap_merge(
                boxes=candi,
                threshold=threshold,
            )

            for _nms in nms_result:
                ensamble_bbox.extend(_nms)
                            
    
    return ensamble_bbox


# %%
num_of_img = 100
min_confidence = 0.3
threshold = 0.8
for d_name in ["overlap_merge"]:#,"55_SwinL.csv","64_yolo.csv",]:
    
    with wandb.init(
        entity="passion-ate", project='images-with-bbox',
        group=f"al1-overlap-highconf-merge-min_conf-{min_confidence:.4f}-iou-thr-{threshold:.4f}",name=d_name
    ):
        te_image_prefix = "../dataset/"
        for im in tqdm(te_dict.images[:num_of_img]):
            im_name = te_image_prefix + im.file_name
            pil_img = Image.open(im_name)

            if d_name in ["55_SwinL.csv","64_yolo.csv",]:
                df = pd.read_csv(f"./test_results/{d_name}")
                splited_anno = df[df["image_id"]==im.file_name]["PredictionString"].to_list()[0].split(" ")
            else:
                splited_anno = get_all_cls_ensamble_bbox(im, min_confidence, threshold)
            
            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(pil_img)
            ax.set_title(d_name)
            for id_idx in range(10):

                for i in range(len(splited_anno)//6):
                    bbox = splited_anno[i*6:(i+1)*6]
                    
                    category_id, confidence  = int(bbox[0]), float(bbox[1])
                    color = palette[category_id]
                    class_name = classes[category_id]

                    x0, y0, x1, y1 = list(map(float, bbox[2:6]))
                    if category_id == id_idx and confidence > min_confidence:
                        rect = patches.Rectangle(
                            xy=(x0, y0),
                            width=abs(x1-x0), height=abs(y1-y0),
                            linewidth=2, edgecolor=color, facecolor='none', alpha=1,#confidence,
                        )
                        ax.add_patch(rect)
                        ax.text(
                            x=x0, y=y0, s=f"{class_name}-{confidence:.2f}",alpha=1,#confidence,
                            bbox=dict(facecolor=color, alpha=confidence, boxstyle= "Round, pad=0.0", edgecolor=color)
                        )
            wandb.log({'image':fig})
            plt.cla()
            plt.clf()
            plt.close()
            del ax
            del fig


# %%



