import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import cv2


def load_model_with_state_dict(model_path, num_classes=2):
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # 初始化模型结构
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # 获取特征输入维度并重新定义预测器
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 加载模型参数
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)     # strict=False 参数允许跳过模型中不匹配的 key, 忽略分类器层的细微差异。
    model.eval()  # 设置为评估模式
    return model

def predict_ppt_regions(model, image_path, threshold=0.8):
    """将图片输入模型，并从模型输出中提取预测的边界框
    
    Args:
        threshold(float): 置信度阈值，用于筛选出置信度较高的边界框。
    Returns:
        selected_boxes: 预测出的PPT区域坐标。
    """
    # 预处理图片
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image)

    # 模型推理
    with torch.no_grad():
        predictions = model([image_tensor])

    # 提取预测结果
    boxes = predictions[0]['boxes'].cpu().numpy()
    print("边界框坐标: ", boxes)
    scores = predictions[0]['scores'].cpu().numpy()
    print("置信度得分: ", scores)

    # 过滤出高置信度的边界框
    selected_boxes = boxes[scores >= threshold]
    return selected_boxes

def visualize_results(image_path, boxes, output_path=None):
    """使用 OpenCV 将预测的边界框绘制在图片上并显示结果"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image at {image_path}.")
        return
    
    for box in boxes:
        # 提取边界框坐标
        x1, y1, x2, y2 = map(int, box)
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示图片
    cv2.imshow("Detected PPT Regions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 如果提供了输出路径，则保存图片
    if output_path:
        cv2.imwrite(output_path, image)

def save_cropped_ppt_regions(image_path, boxes, output_dir="cropped_ppt"):
    """将检测出的 PPT 区域裁剪并保存"""
    image = cv2.imread(image_path)
    os.makedirs(output_dir, exist_ok=True)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]  # 裁剪PPT区域
        output_path = os.path.join(output_dir, f"ppt_region_{i}.jpg")
        cv2.imwrite(output_path, cropped)


if __name__ == '__main__':
    # 加载模型
    model_path = "faster_rcnn_ppt.pth"
    # model = get_trained_model(model_path, num_classes=2)
    # model = torch.load(model_path)
    # model.eval()  # 设置为评估模式
    model = load_model_with_state_dict(model_path, num_classes=2)

    # 推理并显示结果
    image_path = "data\\20240922172937.jpg"  # 要检测的图片路径
    boxes = predict_ppt_regions(model, image_path, threshold=0.6)
    # print("boxes0: ", boxes)

    # 可视化结果
    visualize_results(image_path, boxes, output_path="output_with_boxes.jpg")

    # 保存裁剪后的PPT区域
    # save_cropped_ppt_regions(image_path, boxes, output_dir="cropped_ppt")


