"""
python 3.10.0
torch 2.4.0
torchvision 0.19.0
"""
import torch
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import json
import numpy as np
import cv2
from PIL import Image

## 继续训练用到的库
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

class LabelmeDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        self.img_dir = img_dir      # 图片路径
        self.annotation_dir = annotation_dir  # 标注文件路径
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.annotations = list(sorted(os.listdir(annotation_dir)))

    def __getitem__(self, idx):
        # 加载图片和标注文件
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotations[idx])

        img = Image.open(img_path).convert("RGB")

        # 解析 JSON 文件，获取边界框和分割掩码信息
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        boxes = []
        labels = []
        masks = []

        for shape in annotation['shapes']:
            if shape['label'] == 'ppt':
                # 获取多边形的坐标
                points = np.array(shape['points'])
                xmin, ymin = points.min(axis=0)
                xmax, ymax = points.max(axis=0)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # 'ppt' 标签的 ID 为 1

                # 创建分割掩码
                mask = np.zeros((img.height, img.width), dtype=np.uint8)
                points = points.astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
                masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # 构建 target 字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_mask_rcnn_model(num_classes):
    # # 加载预训练的 Mask R-CNN 模型
    # model = maskrcnn_resnet50_fpn(pretrained=True)

    # 指定权重加载
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights)

    # 修改分类器以输出指定类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 修改掩码预测器以输出指定类别数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# 明确定义 collate_fn 函数
def collate_fn(batch):
    return tuple(zip(*batch))

def start_first_train():
    """第一次训练启动！"""

    # 数据集路径
    img_dir = "data\\raw"
    annotation_dir = "data\\annotate"

    # 加载数据集
    dataset = LabelmeDataset(img_dir, annotation_dir, transforms=T.ToTensor())
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    # 使用 DataLoader 时指定 collate_fn
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)


    # 获取模型
    num_classes = 2  # 假设两类：背景 + PPT
    model = get_mask_rcnn_model(num_classes)

    # 设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {losses.item()}")
    
    torch.save(model.state_dict(), "mask_rcnn_ppt.pth")

## 继续训练用到的其他函数
def continue_training():
    """继续训练已保存的模型"""
    img_dir = "data\\raw1"  # 新数据集路径
    annotation_dir = "data\\annotate1"  # 新标注文件路径
    dataset = LabelmeDataset(img_dir, annotation_dir, transforms=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    num_classes = 2  # 假设两类：背景 + PPT
    model = get_mask_rcnn_model(num_classes)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 加载之前训练的模型权重
    model.load_state_dict(torch.load("mask_rcnn_ppt1.pth"))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10  # 继续训练的轮数
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Continue Training Epoch {epoch+1}, Loss: {losses.item()}")
    
    torch.save(model.state_dict(), "mask_rcnn_ppt1.pth")

if __name__ == '__main__':

    # start_first_train()       # 启动第一次训练！

    continue_training()       # 继续训练
