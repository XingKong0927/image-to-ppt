""" 使用训练好的模型，循环处理每一张图片，提取ppt区域并形成pdf文档
python 3.10.0
torch 2.4.0
torchvision 0.19.0
"""
import os

from test_maskrcnn import get_trained_mask_rcnn_model, predict_ppt_regions_with_min_bounding_box, save_cropped_ppt_regions
from tools.images_to_file import images_to_pdf, images_to_ppt

def to_ppt_image(img_dir, cropped_dir):
    """批量处理照片并保存图片到文件夹中"""
    # 加载模型
    model_path = "mask_rcnn_ppt1.pth"
    model = get_trained_mask_rcnn_model(model_path, num_classes=2)

    imgs = sorted(os.listdir(img_dir))  # 遍历图片文件夹中的所有图片文件

    # 处理每张图片
    for count, img_name in enumerate(imgs):
        image_path = os.path.join(img_dir, img_name)
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # 跳过非图片文件

        print(f"Processing No.{count+1}: {image_path}")

        # 预测并生成最小外接矩形
        boxes = predict_ppt_regions_with_min_bounding_box(model, image_path, threshold=0.8, epsilon_factor=0.07)

        # 保存裁剪后的PPT区域
        save_cropped_ppt_regions(image_path, boxes, output_dir=cropped_dir)

if __name__ == '__main__':
    img_dir = "data\\raw1"    # 待处理照片路径

    # 创建裁剪区域输出文件夹
    cropped_dir = "data\\cropped_ppt1"
    os.makedirs(cropped_dir, exist_ok=True)

    # 应用训练好的maskrcnn模型预测并裁剪图片，并保存到文件夹中
    to_ppt_image(img_dir, cropped_dir)

    # 使用裁剪后的图片生成PDF
    # images_to_pdf(cropped_dir, output_pdf_path="output_ppt.pdf")

    # 使用裁剪后的图片生成PPT
    images_to_ppt(cropped_dir, output_ppt_path="output_ppt1.pptx", slide_size="widescreen")

