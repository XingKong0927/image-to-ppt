import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image


class LabelmeDataset(Dataset):
    """数据集类，用来读取图片及其对应的标注文件，并将这些数据转换为 PyTorch 的格式供模型使用。
    """
    def __init__(self, img_dir, annotation_dir, transforms=None):
        self.img_dir = "data\\raw"  # 图片路径
        self.annotation_dir = "data\\annotate"  # 标注文件路径
        self.transforms = transforms  # 图像增强变换
        self.imgs = list(sorted(os.listdir(img_dir)))  # 所有图片文件
        self.annotations = list(sorted(os.listdir(annotation_dir)))  # 所有JSON标注文件

    def __getitem__(self, idx):
        # 加载图片和标注
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # 读取对应的标注文件
        annotation_path = os.path.join(self.annotation_dir, self.annotations[idx])
        
        # 解析JSON文件，获取边界框信息
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # 解析标注文件中的边界框和标签
        boxes = []
        labels = []
        areas = []      # 用于存储每个边界框的面积
        for shape in annotation['shapes']:
            if shape['label'] == 'ppt':
                # 获取矩形边界框 (xmin, ymin, xmax, ymax)
                points = shape['points']
                xmin = min([p[0] for p in points])
                ymin = min([p[1] for p in points])
                xmax = max([p[0] for p in points])
                ymax = max([p[1] for p in points])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # 假设所有对象都是同一类别，例如PPT
                # 计算并添加面积
                areas.append((xmax - xmin) * (ymax - ymin))

        # 转换为PyTorch张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # 生成target字典，包含boxes, labels, 和 area
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = areas  # 将计算好的面积赋值给area字段
        target["image_id"] = idx
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)  # 必须字段，假设无拥挤情况

        
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model(num_classes):
    # 加载预训练的Faster R-CNN模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取模型的分类器，并重新定义其输出
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.models.detection import FasterRCNN
    from references.detection.engine import train_one_epoch, evaluate
    from references.detection import utils

    # 数据集路径
    img_dir = "data\\raw"
    annotation_dir = "data\\annotate"

    # 加载数据集
    dataset = LabelmeDataset(img_dir, annotation_dir, transforms=F.to_tensor)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    # 获取模型
    num_classes = 2  # 两类：背景 + PPT
    model = get_model(num_classes)

    # 将模型移动到GPU（如果可用）
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader, device=device)

    torch.save(model.state_dict(), "faster_rcnn_ppt.pth")

