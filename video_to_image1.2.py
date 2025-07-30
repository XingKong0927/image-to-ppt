"""
实现：
1. 从屏幕中人工确定ppt区域；
2. 自动识别ppt是否换页，如换页就将ppt部分截取保存为图片。


1.1改进：使用 mss 替代 ImageGrab 实现多屏幕截图，替代了from PIL import ImageGrab
1.2改进：在程序自动“取”图片时保证电脑不休眠
"""

import os, cv2, time, ctypes, mss
import threading
import numpy as np
import tkinter as tk

from tkinter import messagebox
from skimage.metrics import structural_similarity as ssim

from tools.images_to_file import images_to_ppt

# 初始化变量
prev_frame = None                       # 用于存储前一帧的图像
threshold = 0.98                        # 设置图像差异阈值，如果连续两帧差异小于此值，则认为换页
frame_count = 0                         # 记录帧数，用于调试或跟踪
slide_images = []                       # 存储每一帧提取的PPT图片路径
ppt_area = []                           # PPT区域坐标
is_recording = False                    # 是否正在自动保存图片
current_status = "请“识”别PPT区域"       # 初始状态显示
screen_img = None                       # 屏幕截图
selecting = False                       # 框选状态
start_x, start_y = 0, 0                 # 起始坐标
target_monitor = None                   # 所选屏幕四角坐标

# 确保保存PPT文件的目录存在
cropped_dir = "data/video_ppt_images"
os.makedirs(cropped_dir, exist_ok=True)

# 使用 Tkinter 创建GUI界面
root = tk.Tk()
root.iconbitmap("data/icon.ico")        # 程序图标
root.title("四字成片")

# 定义POINT结构体
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

# 定义显示器信息结构体
class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", ctypes.c_long * 4),
        ("rcWork", ctypes.c_long * 4),
        ("dwFlags", ctypes.c_ulong)
    ]


def get_cursor_position():
    """获取鼠标绝对位置"""
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def get_monitors():
    """获取所有显示器信息"""
    monitors = []
    callback = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, 
                                 ctypes.POINTER(ctypes.c_long), ctypes.POINTER(MONITORINFO))
    
    def monitor_callback(hmonitor, hdc, rect, data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        ctypes.windll.user32.GetMonitorInfoW(hmonitor, ctypes.byref(info))
        # 确保获取完整的显示器信息
        monitors.append({
            "left": info.rcMonitor[0],
            "top": info.rcMonitor[1],
            "right": info.rcMonitor[2],
            "bottom": info.rcMonitor[3],
            "width": info.rcMonitor[2] - info.rcMonitor[0],
            "height": info.rcMonitor[3] - info.rcMonitor[1]
        })
        return 1
    
    # 关键修改：使用正确的API获取所有显示器
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, callback(monitor_callback), 0)
    return monitors

def find_current_monitor(monitors):
    """根据鼠标位置自动选择显示器"""
    # 获取鼠标当前位置
    cursor_x, cursor_y = get_cursor_position()
    
    # 查找包含鼠标光标的显示器
    for monitor in monitors:
        if (monitor["left"] <= cursor_x <= monitor["right"] and
            monitor["top"] <= cursor_y <= monitor["bottom"]):
            return monitor
    
    # 如果找不到，返回第一个显示器
    return monitors[0]

def start_selecting_area(event, x, y, flags, param):
    """鼠标回调函数，记录框选坐标"""
    global ppt_area, screen_img, start_x, start_y, selecting, target_monitor
    
    if not target_monitor:
        return
    
    # 转换为屏幕绝对坐标
    abs_x = x + target_monitor["left"]
    abs_y = y + target_monitor["top"]
    
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
        start_x, start_y = abs_x, abs_y
        selecting = True
        print(f"起始点: ({start_x}, {start_y})")
        
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:  # 左键拖曳
        if selecting:
            # 创建临时图像用于实时显示框选区域
            temp_img = screen_img.copy()
            
            # 绘制实时框选矩形
            cv2.rectangle(temp_img, 
                         (start_x - target_monitor["left"], start_y - target_monitor["top"]), 
                         (x, y),
                         (0, 255, 0), 2)
            cv2.imshow('Select PPT Area (ESC to cancel)', temp_img)
    
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        if selecting:
            end_x, end_y = abs_x, abs_y
            selecting = False
            
            # 确保起点在左上，终点在右下
            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)
            
            # 保存框选区域坐标
            ppt_area = [(x1, y1), (x2, y2)]
            print(f"框选区域: ({x1}, {y1}) - ({x2}, {y2})")
            
            # 绘制最终框选区域
            temp_img = screen_img.copy()
            cv2.rectangle(temp_img, 
                         (x1 - target_monitor["left"], y1 - target_monitor["top"]), 
                         (x2 - target_monitor["left"], y2 - target_monitor["top"]),
                         (0, 0, 255), 2)
            cv2.imshow('Select PPT Area (ESC to cancel)', temp_img)

# 按“识”按钮开始手动选择PPT区域
def start_selecting_area_gui():
    """主函数：启动多屏幕区域选择界面"""
    global ppt_area, screen_img, selecting, target_monitor, current_status
    
    # 初始化变量
    ppt_area = []        # 清空选择区域
    selecting = False
    
    # 1. 获取所有显示器信息
    monitors = get_monitors()
    if not monitors:
        # 如果获取失败，使用虚拟屏幕作为后备方案
        user32 = ctypes.windll.user32
        monitors = [{
            "left": user32.GetSystemMetrics(76),
            "top": user32.GetSystemMetrics(77),
            "right": user32.GetSystemMetrics(76) + user32.GetSystemMetrics(78),
            "bottom": user32.GetSystemMetrics(77) + user32.GetSystemMetrics(79),
            "width": user32.GetSystemMetrics(78),
            "height": user32.GetSystemMetrics(79)
        }]
    
    # 2. 根据鼠标位置自动选择显示器
    target_monitor = find_current_monitor(monitors)
    print(f"自动选择显示器: {target_monitor['width']}x{target_monitor['height']} "
          f"@ ({target_monitor['left']},{target_monitor['top']})")
    
    # 3. 截取目标屏幕区域
    with mss.mss() as sct:
        # 创建mss兼容的显示器信息格式
        monitor_dict = {
            "left": target_monitor["left"],
            "top": target_monitor["top"],
            "width": target_monitor["width"],
            "height": target_monitor["height"]
        }
        
        # 捕获屏幕图像
        sct_img = sct.grab(monitor_dict)
        
        # 转换为OpenCV格式
        screen_img = np.array(sct_img)
        screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
    
    # 4. 创建窗口并定位到目标屏幕
    window_name = 'Select PPT Area (ESC to cancel)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 关键步骤：先定位窗口再设成全屏
    cv2.moveWindow(window_name, target_monitor["left"], target_monitor["top"])
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # 5. 设置鼠标回调
    cv2.setMouseCallback(window_name, start_selecting_area)
    
    # 显示窗口
    cv2.imshow(window_name, screen_img)
    
    print("请在屏幕上框选PPT区域，按ESC取消")
    
    # 等待用户框选或取消
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC键退出
            ppt_area = []  # 清空选择
            break
        if ppt_area:  # 已选择区域
            break
            
    cv2.destroyAllWindows()

    if len(ppt_area) == 2:
        current_status = f"PPT区域已选定，请提“取”图片吧~"  # 更新状态
    else:
        current_status = f"请“识”别PPT区域"                     # 初始状态显示
    enable()                # 显示root页面按钮

    return

# 按“取”按钮开始自动截取图片
def start_auto_recording():
    global prev_frame, ppt_area, is_recording, frame_count, current_status, target_monitor

    messagebox.showinfo("提示", "按'止'键暂停截取PPT图片。")

    if ppt_area is None or len(ppt_area) != 2:
        messagebox.showerror("错误", "请先选择PPT区域！")
        return

    if is_recording:
        messagebox.showinfo("提示", "正在自动保存图片中...")
        return

    is_recording = True
    slide_images.clear()  # 清空之前保存的图片
    prev_frame = None     # 重置对比帧缓存
    frame_count = 0       # 重置帧数计数器

    current_status = "开始提取PPT图片..."  # 更新状态
    disable(step=2)               # 锁定root页面按钮

    # 启动一个新线程来处理自动提取
    recording_thread = threading.Thread(target=record_ppt, args=(target_monitor,))
    recording_thread.start()


# 录制PPT区域并计算相似度
def record_ppt(target_monitor):  # 添加参数接收目标显示器信息
    global prev_frame, ppt_area, is_recording, frame_count, current_status

    # 调用系统API阻止休眠：新的常量定义
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    # 定义Windows API函数
    set_thread_exec_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_thread_exec_state.argtypes = [ctypes.wintypes.ULONG]
    set_thread_exec_state.restype = ctypes.wintypes.ULONG

    try:
        # 阻止系统休眠
        set_thread_exec_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

        with mss.mss() as sct:
            # 根据传入的显示器信息创建截图区域
            monitor_dict = {
                "left": target_monitor["left"],
                "top": target_monitor["top"],
                "width": target_monitor["width"],
                "height": target_monitor["height"]
            }

            while is_recording:
                sct_img = sct.grab(monitor_dict)        # 对目标显示器截图
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 修正坐标转换逻辑（绝对坐标转显示器相对坐标）
                x1 = ppt_area[0][0] - target_monitor["left"]
                y1 = ppt_area[0][1] - target_monitor["top"]
                x2 = ppt_area[1][0] - target_monitor["left"]
                y2 = ppt_area[1][1] - target_monitor["top"]

                # 添加边界检查确保不越界
                height, width = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                ppt_frame = frame[y1:y2, x1:x2]

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

    finally:
        # 恢复系统默认电源设置
        set_thread_exec_state(ES_CONTINUOUS)

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
