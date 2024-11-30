# image-to-ppt
初心：我们在参加会议的时候，有时会想把一些好的PPT备份下来，很多参会情况下只能自己拍摄照片。后续的照片处理（image——>PPT）的过程就很痛苦了，于是此项目诞生了..

功能概述：批量处理照片，识别照片中的PPT部分，自动裁剪+变形，并生成为一个PPT文件

**Done：**

1. 模型的训练及增量训练；
2. 照片中PPT区域的识别（如效果较差则需增量训练）;
3. 将PPT区域裁剪出来（XXX目前是以区域的最大最小值裁剪出来的矩形）；
4. 生成PDF/PPT文件。

**Todo：**

1. 在`Mask R-CNN`模型基础上，将识别出来的PPT区域拟合为不规则四边形，以四边形的顶点做图形变换，形成矩形PPT图片；
2. 另一个思路：识别照片中PPT区域的边界，以识别到的边界顶点做图形变换，形成矩形PPT图片；
3. 有时我们会参加线上会议，后续可能增加线上会议的PPT自动抓取功能。

## 使用

1. 将照片下载到“raw”系列文件夹中（**拍照时尽量拍全PPT区域，否则会影响识别准确度！**）；
2. 将批处理工具放到上述文件夹中，使用批处理工具“批-删部分文件名.bat”删除文件中的中文；
3. 使用“labelme”标注照片中的ppt区域，标签设置为“ppt”，并保存标注文件到“annotate”系列文件夹中；
4. 运行“train_maskrcnn.py”主函数，选择“第一次训练”或“继续训练”mask_rcnn模型；
5. 运行“main.py”主函数，批处理待处理照片，并生成ppt文件；
6. 运行“test_maskrcnn.py”可对一张照片进行单独测试。

## 初步开发思路

1. **图像预处理**：使用`Labelme`标注图片，将图片转换为张量。
2. **PPT区域检测**：使用 `Mask R-CNN` 模型检测图片中的PPT区域。
3. **PPT区域裁剪**：根据检测结果裁剪出PPT区域。
4. **透视调整**：如果需要，进行透视变换，将PPT区域调整为标准矩形。
5. **批量处理**：处理多张图片，批量裁剪和调整PPT区域。
6. **生成PDF/PPT**：将处理后的PPT图片合并为一个文件。

## 库

| Name | Version |
| ---- | ------- |
| bzip2 |                     1.0.8 |
| ca-certificates |           2024.9.24 |
| colorama |                  0.4.6 |
| contourpy |                 1.3.0 |
| cycler |                    0.12.1 |
| filelock |                  3.16.1 |
| fonttools |                 4.54.1 |
| fsspec |                    2024.9.0 |
| glib |                      2.78.4 |
| glib-tools |                2.78.4 |
| gst-plugins-base |          1.18.5 |
| gstreamer |                 1.18.5 |
| icu |                       58.2 |
| jinja2 |                    3.1.4 |
| jpeg |                      9e |
| kiwisolver |                1.4.7 |
| krb5 |                      1.19.4 |
| libclang |                  14.0.6 |
| libclang13 |                14.0.6 |
| libffi |                    3.4.4 |
| libglib |                   2.78.4 |
| libiconv |                  1.16 |
| libogg |                    1.3.5 |
| libpng |                    1.6.39 |
| libvorbis |                 1.3.7 |
| lz4-c |                     1.9.4 |
| markupsafe |                3.0.1 |
| matplotlib |                3.9.2 |
| mpmath |                    1.3.0 |
| networkx |                  3.3 |
| numpy |                     1.26.4 |
| openssl |                   1.1.1w |
| packaging |                 24.1 |
| pcre2 |                     10.42 |
| pillow |                    10.4.0 |
| pip |                       24.2 |
| ply |                       3.11 |
| pycocotools |               2.0.8 |
| pyparsing |                 3.1.4 |
| pyqt |                      5.15.10 |
| pyqt5-sip |                 12.13.0 |
| **python** |                    3.10.0 |
| python-dateutil |           2.9.0.post0 |
| pyyaml |                    6.0.2 |
| qt-main |                   5.15.2 |
| qtpy |                      2.4.1 |
| setuptools |                75.1.0 |
| sip |                       6.7.12 |
| six |                       1.16.0 |
| sqlite |                    3.45.3 |
| sympy |                     1.13.3 |
| tk |                        8.6.14 |
| tomli |                     2.0.1 |
| **torch** |                     2.4.0 |
| **torchvision** |               0.19.0 |
| typing-extensions |         4.12.2 |
| tzdata |                    2024b |
| vc |                        14.40 |
| vs2015_runtime |            14.40.33807 |
| wheel |                     0.44.0 |
| xz |                        5.4.6 |
| zlib |                      1.2.13 |
| zstd |                      1.5.6 |

## 小知识XXX

在目标检测中， Faster R-CNN 等模型默认输出的是矩形边界框。
如果需要检测不规则的四边形，可以使用 实例分割模型，比如 Mask R-CNN，它可以生成像素级的分割掩码，从而捕获不规则的形状。