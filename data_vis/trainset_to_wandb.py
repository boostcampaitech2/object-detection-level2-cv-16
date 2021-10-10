#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import json

import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from easydict import EasyDict as edict


# In[2]:


wandb.login()


# In[3]:


train_path = "/opt/ml/detection/dataset/train.json"
with open(train_path, "r") as f:
    train_json = json.load(f)
print(train_json.keys())
tr_dict = edict(train_json)


# In[4]:


print(tr_dict.images[0].keys())
tr_dict.annotations[0]


# In[16]:


palette = [
    'Chartreuse','BlueViolet','Crimson','DarkOrange','DodgerBlue',
    'Gold','HotPink','SkyBlue','Wheat','Tan',
]
classes=[
    'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]


# In[26]:


wandb.init(
    entity="passion-ate",
    project='images-with-bbox',
    name="general-trash-ori-trainset"
)


# In[25]:



tr_image_prefix = "/opt/ml/detection/dataset/"

for im in tr_dict.images[:100]:
    log_it = False
    fig, ax = plt.subplots(figsize=(8, 8))
    im_name = tr_image_prefix + im.file_name
    Img = Image.open(im_name)
    ax.imshow(Img)
    
    for anno in tr_dict.annotations:
        if anno.image_id == im.id:
            class_id = anno.category_id
            class_name = classes[class_id]
            color = palette[class_id]
            if class_name == 'General trash':
                log_it = True
                x1, y1, w, h = anno.bbox
                rect = patches.Rectangle(
                    xy=(x1, y1),
                    width=w, height=h,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.5,
                )
                ax.add_patch(rect)
                # ax.text(
                #     x=x1, y=y1, s=class_name,
                #     bbox=dict(facecolor=color, alpha=0.4)
                # )
    
    if log_it:
        wandb.log({"General trash": fig})
        time.sleep(0.01)
        plt.cla() # clear axis
        plt.clf() # clear fig
        # plt.show()


# In[ ]:




