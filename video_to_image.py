"""
实现：
1. 从屏幕中人工确定ppt区域；
2. 自动识别ppt是否换页，如换页就将ppt部分截取保存为图片。
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageGrab
import os
import tkinter as tk
from tkinter import messagebox
import time


# 初始化变量
prev_frame = None  # 用于存储前一帧的图像
threshold = 0.5  # 设置图像差异阈值，如果连续两帧差异大于此值，则认为换页
frame_count = 0  # 记录帧数，用于调试或跟踪
slide_images = []  # 存储每一帧提取的PPT图片路径
is_ppt_area_selected = False  # 标志是否已选择PPT区域
ppt_area = None  # PPT区域坐标
is_recording = False  # 是否正在自动保存图片

# 确保保存PPT文件的目录存在
output_dir = "output_ppt_images"
os.makedirs(output_dir, exist_ok=True)

# 使用 Tkinter 创建GUI界面
root = tk.Tk()
root.title("PPT自动提取工具")

# 鼠标左键按下时开始选择区域，右键撤销上一个选择点
def start_selecting_area(event, x, y, flags, param):
    global frame, ppt_area, is_ppt_area_selected, point1, point2

    img2 = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        if len(ppt_area) == 0:
            ppt_area.append(point1)  # 记录第一个点
            print(f"选择区域的起始点：{ppt_area[0]}")
        else:
            ppt_area[0] = point1
            print(f"选择区域的起始点：{ppt_area[0]}")
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        if len(ppt_area) == 1:
            ppt_area.append((x, y))  # 记录第二个点并结束选择
            is_ppt_area_selected = True
            print(f"选择区域的终止点：{ppt_area[1]}")
        else:
            ppt_area[1] = point2
            is_ppt_area_selected = True
            print(f"选择区域的终止点：{ppt_area[1]}")
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)

# 按“识”按钮开始手动选择PPT区域
def start_selecting_area_gui():
    global frame, ppt_area, is_ppt_area_selected
    ppt_area = []  # 清空选择区域

    beg = time.time()
    debug = False
    image = ImageGrab.grab()
    image.save("screen.jpg")

    frame = cv2.imread('screen.jpg')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', start_selecting_area)
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    os.remove('screen.jpg')
    # cv2.destroyAllWindows()

# 按“取”按钮开始自动截取图片
def start_auto_recording():
    global prev_frame, ppt_area, is_recording, frame_count

    messagebox.showinfo("提示", "按q键停止截取ppt图片。")

    if ppt_area is None or len(ppt_area) != 2:
        messagebox.showerror("错误", "请先选择PPT区域！")
        return

    if is_recording:
        messagebox.showinfo("提示", "正在自动保存图片中...")
        return

    is_recording = True
    slide_images.clear()  # 清空之前保存的图片

    print("开始自动保存PPT图片...")
    
    while True:
        # 截取当前屏幕
        screen = np.array(ImageGrab.grab())  # 截取整个屏幕
        frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)  # 转换为OpenCV格式

        # 获取指定区域内的PPT部分
        ppt_frame = frame[ppt_area[0][1]:ppt_area[1][1], ppt_area[0][0]:ppt_area[1][0]]

        # 将当前帧转换为灰度图，减少计算量
        gray_frame = cv2.cvtColor(ppt_frame, cv2.COLOR_BGR2GRAY)

        # 如果是第一帧，则跳过
        if prev_frame is None:
            prev_frame = gray_frame
            continue

        # 计算当前帧与前一帧的结构相似度（SSIM）
        similarity_index, _ = ssim(prev_frame, gray_frame, full=True)

        # 如果当前帧与前一帧的相似度低于设定阈值，认为 PPT 已经换页
        if similarity_index < threshold:
            print(f"Page change detected at frame {frame_count}")

            # 保存当前帧为图片
            image_filename = f"{output_dir}/extracted_ppt_slide_{frame_count}.png"
            cv2.imwrite(image_filename, ppt_frame)
            slide_images.append(image_filename)  # 将图片路径保存到列表中
            print(f"Saved slide as {image_filename}")

        # 更新上一帧为当前帧，为下一次对比做准备
        prev_frame = gray_frame

        # 增加帧数计数器（用于调试）
        frame_count += 1

        # # 显示视频帧（可选）
        # cv2.imshow("PPT Slide Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        # 按‘q’退出录制模式
        if key == ord('q'):
            break

    is_recording = False
    cv2.destroyAllWindows()
    print("自动保存完成。")


# 创建界面按钮
select_button = tk.Button(root, text="识", command=start_selecting_area_gui, height=2, width=10)
select_button.pack(pady=10)

record_button = tk.Button(root, text="取", command=start_auto_recording, height=2, width=10)
record_button.pack(pady=10)

# 启动 Tkinter 窗口
root.mainloop()
