"""
python 3.10.0
torch 2.4.0
torchvision 0.19.0

在目标检测中， Faster R-CNN 等模型默认输出的是矩形边界框。
如果需要检测不规则的四边形，可以使用 实例分割 模型，比如 Mask R-CNN，
它可以生成像素级的分割掩码，从而捕获不规则的形状。

"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import json
import numpy as np
from PIL import Image
import cv2


class LabelmeDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        self.img_dir = "data\\raw"      # 图片路径
        self.annotation_dir = "data\\annotate"  # 标注文件路径
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
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

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

