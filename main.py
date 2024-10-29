""" 使用训练好的模型，循环处理每一张图片，提取ppt区域并形成pdf文档
python 3.10.0
torch 2.4.0
torchvision 0.19.0
"""
import os
from PIL import Image

from test_maskrcnn import get_trained_mask_rcnn_model, predict_ppt_regions_with_min_bounding_box, save_cropped_ppt_regions


def to_ppt_image(img_dir, cropped_dir):
    """批量处理照片并保存图片到文件夹中"""
    # 加载模型
    model_path = "mask_rcnn_ppt.pth"
    model = get_trained_mask_rcnn_model(model_path, num_classes=2)

    imgs = sorted(os.listdir(img_dir))  # 遍历图片文件夹中的所有图片文件

    # 处理每张图片
    for img_name in imgs:
        image_path = os.path.join(img_dir, img_name)
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # 跳过非图片文件

        print(f"Processing {image_path}")

        # 预测并生成最小外接矩形
        boxes = predict_ppt_regions_with_min_bounding_box(model, image_path, threshold=0.8, epsilon_factor=0.07)

        # 保存裁剪后的PPT区域
        save_cropped_ppt_regions(image_path, boxes, output_dir=cropped_dir)

def image_to_pdf(output_dir, output_pdf_path):
    """保存图片为PDF文档
    
    Arg:
        output_dir: 裁剪后的图片文件夹
        output_pdf_path: 生成的PDF文件路径
    """
    # 检查文件夹是否为空
    cropped_images = list(sorted(os.listdir(output_dir)))
    if cropped_images:
        # 生成完整路径列表
        image_list = [Image.open(os.path.join(output_dir, img)).convert("RGB") for img in cropped_images]
        # 将第一张图片保存为 PDF，其他图片作为追加
        image_list[0].save(output_pdf_path, save_all=True, append_images=image_list[1:])
        print(f"PPT区域PDF已生成: {output_pdf_path}")
    else:
        print(f"未找到任何图片文件在文件夹 {output_dir} 中。")


if __name__ == '__main__':
    img_dir = "data"    # 待处理照片路径

    # 创建裁剪区域输出文件夹
    cropped_dir = "cropped_ppt"
    os.makedirs(cropped_dir, exist_ok=True)

    # 应用训练好的maskrcnn模型预测并裁剪图片，并保存到文件夹中
    to_ppt_image(img_dir, cropped_dir)

    # 使用裁剪后的图片生成PDF
    image_to_pdf(cropped_dir, output_pdf_path="output_ppt.pdf")

