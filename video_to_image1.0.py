"""
实现：
1. 从屏幕中人工确定ppt区域；
2. 自动识别ppt是否换页，如换页就将ppt部分截取保存为图片。(待验证！！！！！！！！！！！)

bug：多屏幕时只能用于主屏幕
"""

import os, cv2, time
import threading
import numpy as np
import tkinter as tk

from tkinter import messagebox
from PIL import ImageGrab
from skimage.metrics import structural_similarity as ssim

from tools.images_to_file import images_to_ppt

# 初始化变量
prev_frame = None                       # 用于存储前一帧的图像
threshold = 0.98                        # 设置图像差异阈值，如果连续两帧差异小于此值，则认为换页
frame_count = 0                         # 记录帧数，用于调试或跟踪
slide_images = []                       # 存储每一帧提取的PPT图片路径
is_ppt_area_selected = False            # 标志是否已选择PPT区域
ppt_area = None                         # PPT区域坐标
is_recording = False                    # 是否正在自动保存图片
current_status = "请“识”别PPT区域"       # 初始状态显示

# 确保保存PPT文件的目录存在
cropped_dir = "data/video_ppt_images"
os.makedirs(cropped_dir, exist_ok=True)

# 使用 Tkinter 创建GUI界面
root = tk.Tk()
root.iconbitmap("data/icon.ico")        # 程序图标
root.title("四字成片")

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
        cv2.imshow('Please select the PPT area on the screen and close this page directly after selection~', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('Please select the PPT area on the screen and close this page directly after selection~', img2)
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
        cv2.imshow('Please select the PPT area on the screen and close this page directly after selection~', img2)

# 按“识”按钮开始手动选择PPT区域
def start_selecting_area_gui():
    global prev_frame, frame, ppt_area, is_ppt_area_selected, current_status
    
    ppt_area = []       # 清空选择区域
    prev_frame = None                       # 置空前一帧的图像

    image = ImageGrab.grab()
    image.save("screen.jpg")

    frame = cv2.imread('screen.jpg')
    cv2.namedWindow('Please select the PPT area on the screen and close this page directly after selection~')

    # 更新状态显示标签
    current_status = f"请在弹出页面中框选PPT区域"  # 更新状态
    disable()               # 锁定root页面按钮

    cv2.moveWindow('Please select the PPT area on the screen and close this page directly after selection~',0,0)        # 弹出窗口位置
    cv2.setMouseCallback('Please select the PPT area on the screen and close this page directly after selection~', start_selecting_area)
    cv2.imshow('Please select the PPT area on the screen and close this page directly after selection~', frame)

    cv2.waitKey(0)
    os.remove('screen.jpg')

    if len(ppt_area) == 2:
        current_status = f"PPT区域已选定，请提“取”图片吧~"  # 更新状态
    else:
        current_status = f"请“识”别PPT区域"                     # 初始状态显示
    enable()                # 显示root页面按钮

    return

# 按“取”按钮开始自动截取图片
def start_auto_recording():
    global prev_frame, ppt_area, is_recording, frame_count, current_status

    messagebox.showinfo("提示", "按'止'键暂停截取PPT图片。")

    if ppt_area is None or len(ppt_area) != 2:
        messagebox.showerror("错误", "请先选择PPT区域！")
        return

    if is_recording:
        messagebox.showinfo("提示", "正在自动保存图片中...")
        return

    is_recording = True
    slide_images.clear()  # 清空之前保存的图片

    current_status = "开始提取PPT图片..."  # 更新状态
    disable(step=2)               # 锁定root页面按钮

    # 启动一个新线程来处理自动提取
    recording_thread = threading.Thread(target=record_ppt)
    recording_thread.start()


# 录制PPT区域并计算相似度
def record_ppt():
    global prev_frame, ppt_area, is_recording, frame_count, current_status

    while is_recording:
        # 截取当前屏幕
        screen = np.array(ImageGrab.grab())  # 截取整个屏幕
        frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)  # 转换为OpenCV格式

        # 获取指定区域内的PPT部分
        ppt_frame = frame[ppt_area[0][1]:ppt_area[1][1], ppt_area[0][0]:ppt_area[1][0]]

        # 将当前帧转换为灰度图，减少计算量
        gray_frame = cv2.cvtColor(ppt_frame, cv2.COLOR_BGR2GRAY)

        # 如果是第一帧，则仅保存图片
        if prev_frame is None:
            prev_frame = gray_frame
            image_filename = f"{cropped_dir}/video_slide_{time.strftime('%Y%m%d_%H%M%S')}_{frame_count}.png"
            cv2.imwrite(image_filename, ppt_frame)
            continue

        # 计算当前帧与前一帧的结构相似度（SSIM）
        similarity_index, _ = ssim(prev_frame, gray_frame, full=True)
        # print("当前帧与前一帧相似度：{}".format(similarity_index))

        # 更新状态显示标签
        current_status = f"当前帧与前一帧相似度: {similarity_index:.4f}"  # 更新状态
        disable(step=2)               # 锁定root页面按钮

        # 如果当前帧与前一帧的相似度低于设定阈值，认为 PPT 已经换页
        if similarity_index < threshold:
            print(f"Page change detected at frame {frame_count}")

            # 保存当前帧为图片
            image_filename = f"{cropped_dir}/video_slide_{time.strftime('%Y%m%d_%H%M%S')}_{frame_count}.png"
            cv2.imwrite(image_filename, ppt_frame)
            slide_images.append(image_filename)  # 将图片路径保存到列表中
            print(f"Saved slide as {image_filename}")

        # 更新上一帧为当前帧，为下一次对比做准备
        prev_frame = gray_frame

        # 增加帧数计数器（用于调试）
        frame_count += 1

        # 每过1秒查看一下屏幕
        time.sleep(1)

    enable()            # 显示root页面按钮
    return

# 使用Tkinter定期更新状态显示
def update_status(update_time=500):
    """使用Tkinter定期更新状态显示，默认每500毫秒更新一次状态"""
    status_label.config(text=current_status)
    if update_time != 0:
        root.after(update_time, update_status)
    else:
        root.update_idletasks()  # 刷新GUI状态标签

# 按“止”按钮停止保存图片
def stop_recording():
    global is_recording, current_status

    if not is_recording:
        messagebox.showinfo("提示", "当前没有提取PPT图片！")
        return
    else:
        is_recording = False

        current_status = "暂停提取PPT图片..."  # 更新状态
        update_status(update_time=0)
        return

# 按“合”按钮合成PPT
def combine_images_to_ppt():
    """"将提取到的图片合成为PPT"""

    global is_recording, current_status

    if is_recording:
        messagebox.showinfo("警告", "当前正在提取PPT图片，提取已暂停，请重新操作！")
        is_recording = False
        
        current_status = "暂停提取PPT图片..."  # 更新状态
        update_status(update_time=0)
        return
    else:
        current_status = "正在合成PPT文件，请稍后..."   # 更新状态
        disable()                       # 锁定root页面按钮
        # 使用裁剪后的图片生成PPT
        images_to_ppt(cropped_dir, output_ppt_path="video_output_ppt.pptx", slide_size="widescreen")
        current_status = "已合成PPT文件，请及时另存"    # 更新状态
        enable()
        return

# # 设置窗口最大尺寸
# root.maxsize(800, 600)        # 宽，高

# 创建界面按钮
select_button = tk.Button(root, text="识", font=("黑体", 18), command=start_selecting_area_gui, height=2, width=10, state=tk.NORMAL)
select_button.grid(row=0, column=0)

record_button = tk.Button(root, text="取", font=("黑体", 18), command=start_auto_recording, height=2, width=10, state=tk.NORMAL)
record_button.grid(row=0, column=1)

stop_button = tk.Button(root, text="止", font=("黑体", 18), command=stop_recording, height=2, width=10, state=tk.NORMAL)
stop_button.grid(row=1, column=0)

combine_button = tk.Button(root, text="合", font=("黑体", 18), command=combine_images_to_ppt, height=2, width=10, state=tk.NORMAL)
combine_button.grid(row=1, column=1)

# 创建状态显示的标签
status_label = tk.Label(root, text=f"{current_status}", font=("Arial", 12))
status_label.grid(row=2, column=0, columnspan=2)

def disable(step = 1):
    """根据不同进展隐藏root页面按钮"""
    if step == 1:           # 识，合
        select_button.grid_forget()
        record_button.grid_forget()
        stop_button.grid_forget()
        combine_button.grid_forget()
    if step == 2:           # 取
        select_button.grid_forget()
        record_button.grid_forget()
        combine_button.grid_forget()
    update_status(update_time=0)
    return

def enable():
    """显示root页面按钮"""
    select_button.grid(row=0, column=0)
    record_button.grid(row=0, column=1)
    stop_button.grid(row=1, column=0)
    combine_button.grid(row=1, column=1)
    update_status(update_time=0)
    return

# # 置顶显示
# root.attributes('-topmost', 'true')

# 启动 Tkinter 窗口
root.mainloop()
