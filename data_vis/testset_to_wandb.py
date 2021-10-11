#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import json

import wandb
import pandas as pd
from PIL import Image
from easydict import EasyDict as edict
import plotly.express as px

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('int', type=int, help='an integer')
args = parser.parse_args()

# In[2]:


wandb.login()


# In[3]:


test_path = "../dataset/test.json"
with open(test_path, "r") as f:
    test_json = json.load(f)
print(test_json.keys())
te_dict = edict(test_json)

# In[5]:
lb_score = ['mAP_0.647','mAP_0.646','mAP_0.644','mAP_0.394','mAP_0.373','mAP_0.372'][args.int]
df = pd.read_csv(f"../dataset/result/{lb_score}.csv")


# In[6]:


palette = [
    'Chartreuse','BlueViolet','Crimson','DarkOrange','DodgerBlue',
    'Gold','HotPink','SkyBlue','Wheat','Tan',
]
classes=[
    'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]


# In[ ]:


wandb.init(
    entity="passion-ate",
    project='images-with-bbox',
    name=f"lb_score_{lb_score}"
)

te_image_prefix = "../dataset/"

for im in te_dict.images[:30]:
    
    im_name = te_image_prefix + im.file_name
    pil_img = Image.open(im_name)
    
    fig = px.imshow(pil_img, binary_string=False,)
    
    bboxes_all = []
    bboxes_per_cate = {i:[] for i in range(len(classes))}
        
    for anno in df[df["image_id"]==im.file_name]["PredictionString"]:
        splited_anno_str = anno.split(' ')
        for i in range(len(splited_anno_str)//6):
            bbox = splited_anno_str[i*6:(i+1)*6]
            
            category_id, confidence  = int(bbox[0]), float(bbox[1])
            x0, y0, w, h = list(map(float, bbox[2:6]))
            
            # Add buttons that add shapes
            bbox_dict = dict(
                type="rect",
                xref="x", yref="y",
                x0=x0, y0=y0,
                x1=x0+w, y1=y0+h,
                line=dict(color=palette[category_id], width=4),
                opacity=confidence,
            )
            
            bboxes_per_cate[category_id].append(bbox_dict)
            bboxes_all.append(bbox_dict)
        
    buttons = [
        dict(
            label=classes[i],
            method="relayout",
            args=["shapes", bboxes_per_cate[i]]
        ) for i in range(10)
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="None",
                        method="relayout",
                        args=["shapes", []]),
                    *buttons,
                    dict(label="All_Category",
                        method="relayout",
                        args=["shapes", bboxes_all]),
                    
                ]
            )
        ]
    )
        
    wandb.log({"bbox result": wandb.data_types.Plotly(fig)})


