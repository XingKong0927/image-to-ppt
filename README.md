# image-to-ppt
批量处理照片，识别ppt内容，自动裁剪+变形，并生成为一个ppt

## 安装库

Name                    Version                   Build  Channel
ca-certificates           2024.9.24            haa95532_0    defaults
certifi                   2022.12.7        py37haa95532_0    defaults
openssl                   1.1.1w               h2bbff1b_0    defaults
pip                       22.3.1           py37haa95532_0    defaults
python                    3.7.4                h5263a28_0    defaults
setuptools                65.6.3           py37haa95532_0    defaults
sqlite                    3.45.3               h2bbff1b_0    defaults
vc                        14.40                h2eaa2aa_1    defaults
vs2015_runtime            14.40.33807          h98bb1dd_1    defaults
wheel                     0.38.4           py37haa95532_0    defaults
wincertstore              0.2              py37haa95532_2    defaults

## 思路

1. **图像预处理**：使用`labelme`标注图片，将图片转换为张量。
2. **PPT区域检测**：使用 `Faster R-CNN` 模型检测图片中的PPT区域。
3. **PPT区域裁剪**：根据检测结果裁剪出PPT区域。
4. **透视调整**：如果需要，进行透视变换，将PPT区域调整为标准矩形。
5. **批量处理**：处理多张图片，批量裁剪和调整PPT区域。
6. **生成PDF**：将处理后的PPT图片合并为一个PDF文件。