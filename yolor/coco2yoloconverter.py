import os
import shutil

from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CustomDataset(Dataset):

    def __init__(self, annotation, data_dir, mode):
        super().__init__()
        self.data_dir = data_dir
        # load coco annotation  (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.mode = mode

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        
        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode != 'test':
            ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
            anns = self.coco.loadAnns(ann_ids)

            boxes = np.array([x['bbox'] for x in anns])

            labels = np.array([x['category_id'] for x in anns])

            boxes[:,0] = boxes[:,0] + boxes[:,2] / 2
            boxes[:,1] = boxes[:,1] + boxes[:,3] / 2

            boxes /= 1024

            target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index])}

            data = []
            for i,j in zip(target['boxes'], target['labels']):
                x,y,w,h = i
                data.append([j, x, y, w, h])

            filename = image_info['file_name'][:-4].replace('/', '_')
            data = np.array(data)

            np.savetxt(os.path.join('./yolor_paper/yolor/trash_data_3/labels/' + f'{self.mode}/{filename}.txt'),
                          data,
                          fmt = ['%d', '%f', '%f', '%f', '%f'])

            new_img = Image.fromarray(image)
            new_img.save('./yolor_paper/yolor/trash_data_3/images/' + f'{self.mode}/{filename}.jpg')
            
        else:
            filename = image_info['file_name'].replace('/', '_')
            shutil.copyfile(os.path.join(self.data_dir, image_info['file_name']),
                            os.path.join('./yolor_paper/yolor/trash_data_3/images/test', filename))
    
    def __len__(self):
        return len(self.coco.getImgIds())
    
if __name__ == '__main__':
    data_dir = './dataset'
    
    annotation = data_dir + '/train_v3_f3.json'
    temp = CustomDataset(annotation, data_dir, 'train')
    
    # annotation = data_dir + '/valid_v3_f3.json'
    # temp = CustomDataset(annotation, data_dir, 'valid')

    #annotation = data_dir + '/test.json'
    #temp = CustomDataset(annotation, data_dir, 'test')
    
    for i in range(len(temp)):
        print(temp[i])