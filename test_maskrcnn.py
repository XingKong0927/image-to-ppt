import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T

import numpy as np
import cv2
from PIL import Image


def get_trained_mask_rcnn_model(model_path, num_classes=2):
    # 加载预训练的 Mask R-CNN 模型
    model = maskrcnn_resnet50_fpn(pretrained=False)
    
    # 获取分类器的输入特征数并重新定义预测器
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取分割掩码的输入特征数并重新定义掩码预测器
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # 加载模型参数
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_ppt_regions_with_mask(model, image_path, threshold=0.8):
    """预测得到不规则多边形曲线"""
    # 读取和预处理图片
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image)

    model.eval()
    # 模型推理
    with torch.no_grad():
        predictions = model([image_tensor])

    # 提取预测结果
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    masks = predictions[0]['masks'].cpu().numpy()

    # 筛选高置信度的分割结果
    selected_masks = []
    for i, score in enumerate(scores):
        if score >= threshold:
            selected_masks.append(masks[i, 0] > 0.5)  # 阈值0.5用于二值化mask
    return boxes, selected_masks

def visualize_masks(image_path, masks, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image at {image_path}.")
        return

    for mask in masks:
        # 将 mask 转换为轮廓
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # 最接近的四边形
        # 绘制轮廓
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # # 显示结果
    # cv2.imshow("Detected PPT Regions", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, image)

def predict_ppt_regions_as_quadrilaterals(model, image_path, threshold=0.8):
    """预测得到不规则四边形
    
    Arg:
        
    
    """
    # 加载并预处理图片
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image)

    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])

    # 获取高置信度的掩码
    masks = prediction[0]['masks'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    quadrilateral_contours = []     # 存储所有符合要求的四边形轮廓。
    for i, score in enumerate(scores):
        if score >= threshold:
            mask = (masks[i, 0] > 0.5).astype(np.uint8)  # 二值化 mask

            # 查找掩码的轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 计算轮廓的凸包
                hull = cv2.convexHull(contour)
                quadrilateral_contours.append(hull)

            # ### 最小外接矩形
            # for contour in contours:
            #     # 获取最小外接矩形并转换为四边形顶点
            #     rect = cv2.minAreaRect(contour)
            #     box = cv2.boxPoints(rect)
            #     box = np.int0(box)  # 将坐标转换为整数
            #     quadrilateral_contours.append(box)

    return quadrilateral_contours

def predict_ppt_regions_with_min_bounding_box(model, image_path, threshold=0.8, epsilon_factor=0.02):
    """预测得到最小外接矩形
    
    Arg:
        epsilon_factor: 决定多边形近似的精度。较小的值会生成更贴合的四边形，较大的值会生成更粗略的四边形。
    
    """
    # 加载并预处理图片
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image)

    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])

    # 获取高置信度的掩码
    masks = prediction[0]['masks'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    quadrilateral_contours = []     # 存储所有符合要求的四边形轮廓。
    for i, score in enumerate(scores):
        if score >= threshold:
            mask = (masks[i, 0] > 0.5).astype(np.uint8)  # 二值化 mask

            # 查找掩码的轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ### 最小外接矩形
            for contour in contours:
                # 获取最小外接矩形并转换为四边形顶点
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)  # 将坐标转换为整数
                quadrilateral_contours.append(box)

            # ### 最接近的四边形 - 会有部分PPT分割在区域外
            # for contour in contours:
            #     # 使用多边形近似算法，将轮廓简化为四边形
            #     epsilon = epsilon_factor * cv2.arcLength(contour, True)
            #     approx = cv2.approxPolyDP(contour, epsilon, True)       # 最接近的四边形

            #     # 仅保留四边形的轮廓
            #     if len(approx) == 4:
            #         quadrilateral_contours.append(approx)

    return quadrilateral_contours

def visualize_quadrilateral_results(image_path, quadrilateral_contours, output_path=None):
    """将生成的四边形轮廓绘制在原始图片上"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image at {image_path}.")
        return

    # 绘制四边形轮廓
    for contour in quadrilateral_contours:
        cv2.polylines(image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

    # # 显示或保存结果
    # cv2.imshow("Detected PPT Regions as Quadrilaterals", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, image)

def save_cropped_ppt_regions(image_path, boxes, output_dir="cropped_ppt"):
    """将检测出的 PPT 区域裁剪并保存
    
    Arg:
        image_path: 要处理的照片路径
        boxes: ppt在照片中的范围
        output_dir: 输出的文件夹名称
    
    """
    # 使用Pillow加载原始图片
    image = Image.open(image_path)
    # print("image_path: ", image_path)
    imagename = image_path.split("\\")[-1].split(".")[0]

    # 遍历所有检测到的区域
    for i, box in enumerate(boxes):
        # 将四边形坐标转换为 numpy 数组以便计算最小边界框
        box = np.array(box)
        
        # 获取四边形的最小和最大 x, y 值作为边界框
        x1, y1 = box[:, 0].min(), box[:, 1].min()
        x2, y2 = box[:, 0].max(), box[:, 1].max()
        
        # 裁剪图片并保存
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image.save(os.path.join(output_dir, f"ppt_{imagename}.jpg"))

    # print(f"所有PPT区域已成功保存至 {output_dir}")


if __name__ == '__main__':
    # 加载模型
    model_path = "mask_rcnn_ppt.pth"
    model = get_trained_mask_rcnn_model(model_path, num_classes=2)
    
    image_path = "data\\20240922173030.jpg"     # 替换为要检测的图片路径

    # # 预测并生成多边形曲线轮廓
    # boxes, masks = predict_ppt_regions_with_mask(model, image_path, threshold=0.8)
    # # 可视化不规则PPT区域
    # visualize_masks(image_path, masks, output_path="output_with_masks.jpg")

    # 预测并生成最小外接矩形
    boxes = predict_ppt_regions_with_min_bounding_box(model, image_path, threshold=0.8, epsilon_factor=0.07)
    # 可视化结果
    visualize_quadrilateral_results(image_path, boxes, output_path="output_with_min_bounding_box.jpg")

    # # 预测并生成凸包轮廓
    # boxes = predict_ppt_regions_as_quadrilaterals(model, image_path, threshold=0.8)
    # # 可视化结果
    # visualize_quadrilateral_results(image_path, boxes, output_path="output_with_quadrilaterals.jpg")

    # 保存裁剪后的PPT区域
    save_cropped_ppt_regions(image_path, boxes, output_dir="cropped_ppt")



