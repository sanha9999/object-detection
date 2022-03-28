import os
import json
from PIL import Image, ImageFile
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BaekDataset(Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        image_path = '/data/Emotion_data/Validation/image'
        label_path = '/data/Emotion_data/Validation/label'

        self.data_list = glob(image_path + '/*')
        self.label_list = glob(label_path + '/*')
            
        # label map
        self.label_map = {
            '기쁨' : 1,
            '상처' : 2,
            '당황' : 3,
            '분노' : 4,
            '불안' : 5,
            '슬픔' : 6,
            '중립' : 7
        }
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # load images and mask
        img_path = self.data_list[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform  is not None:
            img = self.transform(img)
            
        # filename만 따로 빼서 for문 돌려서 json_list에 있는 것과 비교
        img_name = img_path.split('/')
        mask = {}
        for json_list in self.label_list:
            with open(json_list, 'r') as f:
                json_data = json.load(f)
                for i in range(0, len(json_data)):
                    filename = json_data[i]['filename']
                    if filename == img_name[-1]:
                        mask = json_data[i]
                        
        
        # area : box의 면적으로써 나중에 IOU구하려고 만든거.
        x_min = mask['annot_A']['boxes']['minX']
        x_max = mask['annot_A']['boxes']['maxX']
        y_min = mask['annot_A']['boxes']['minY']
        y_max = mask['annot_A']['boxes']['maxY']
        boxes = [x_min, y_min, x_max, y_max]
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        boxes = boxes.unsqueeze(0)
        
        #area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        
        # label
        label = self.label_map[mask['faceExp_uploader']]
        label = torch.as_tensor(label, dtype=torch.int64)
        # return target
        target = {}
        target["boxes"] = boxes
        target["labels"] = label.unsqueeze(0)
        #target["area"] = area.to(device)
        

        return img, target


def get_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), ])

    dataset = BaekDataset(transform)

    train_size = int(config.train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    dev_size = train_size - int(0.8 * train_size)
    train_size = train_size - dev_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = config.batch_size,
        shuffle= True,
        collate_fn = collate_func
    )

    valid_loader = DataLoader(
        dataset = validation_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = collate_func
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = config.batch_size,
        shuffle = False,
        collate_fn = collate_func
    )

    return train_loader, valid_loader, test_loader

def collate_func(batch):
    return tuple(zip(*batch))