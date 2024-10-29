# image-to-ppt
批量处理照片，识别照片中的ppt部分，自动裁剪+变形，并生成为一个ppt

## 安装库

# Name                    Version                   Build  Channel
bzip2                     1.0.8                h2bbff1b_6    defaults
ca-certificates           2024.9.24            haa95532_0    defaults
colorama                  0.4.6                    pypi_0    pypi
contourpy                 1.3.0                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
filelock                  3.16.1                   pypi_0    pypi
fonttools                 4.54.1                   pypi_0    pypi
fsspec                    2024.9.0                 pypi_0    pypi
glib                      2.78.4               hd77b12b_0    defaults
glib-tools                2.78.4               hd77b12b_0    defaults
gst-plugins-base          1.18.5               h9e645db_0    defaults
gstreamer                 1.18.5               hd78058f_0    defaults
icu                       58.2                 ha925a31_3    defaults
jinja2                    3.1.4                    pypi_0    pypi
jpeg                      9e                   h827c3e9_3    defaults
kiwisolver                1.4.7                    pypi_0    pypi
krb5                      1.19.4               h5b6d351_0    defaults
libclang                  14.0.6          default_hb5a9fac_1    defaults
libclang13                14.0.6          default_h8e68704_1    defaults
libffi                    3.4.4                hd77b12b_1    defaults
libglib                   2.78.4               ha17d25a_0    defaults
libiconv                  1.16                 h2bbff1b_3    defaults
libogg                    1.3.5                h2bbff1b_1    defaults
libpng                    1.6.39               h8cc25b3_0    defaults
libvorbis                 1.3.7                he774522_0    defaults
lz4-c                     1.9.4                h2bbff1b_1    defaults
markupsafe                3.0.1                    pypi_0    pypi
matplotlib                3.9.2                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
networkx                  3.3                      pypi_0    pypi
numpy                     1.26.4                   pypi_0    pypi
openssl                   1.1.1w               h2bbff1b_0    defaults
packaging                 24.1            py310haa95532_0    defaults
pcre2                     10.42                h0ff8eda_1    defaults
pillow                    10.4.0                   pypi_0    pypi
pip                       24.2            py310haa95532_0    defaults
ply                       3.11            py310haa95532_0    defaults
pycocotools               2.0.8                    pypi_0    pypi
pyparsing                 3.1.4                    pypi_0    pypi
pyqt                      5.15.10         py310hd77b12b_0    defaults
pyqt5-sip                 12.13.0         py310h2bbff1b_0    defaults
python                    3.10.0               h96c0403_3    defaults
python-dateutil           2.9.0.post0              pypi_0    pypi
pyyaml                    6.0.2                    pypi_0    pypi
qt-main                   5.15.2               he8e5bd7_8    defaults
qtpy                      2.4.1                    pypi_0    pypi
setuptools                75.1.0          py310haa95532_0    defaults
sip                       6.7.12          py310hd77b12b_0    defaults
six                       1.16.0                   pypi_0    pypi
sqlite                    3.45.3               h2bbff1b_0    defaults
sympy                     1.13.3                   pypi_0    pypi
tk                        8.6.14               h0416ee5_0    defaults
tomli                     2.0.1           py310haa95532_0    defaults
torch                     2.4.0                    pypi_0    pypi
torchvision               0.19.0                   pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
tzdata                    2024b                h04d1e81_0    defaults
vc                        14.40                h2eaa2aa_1    defaults
vs2015_runtime            14.40.33807          h98bb1dd_1    defaults
wheel                     0.44.0          py310haa95532_0    defaults
xz                        5.4.6                h8cc25b3_1    defaults
zlib                      1.2.13               h8cc25b3_1    defaults
zstd                      1.5.6                h8880b57_0    defaults


## 思路

1. **图像预处理**：使用`labelme`标注图片，将图片转换为张量。
2. **PPT区域检测**：使用 `mask R-CNN` 模型检测图片中的PPT区域。
3. **PPT区域裁剪**：根据检测结果裁剪出PPT区域。
4. **透视调整**：如果需要，进行透视变换，将PPT区域调整为标准矩形。
5. **批量处理**：处理多张图片，批量裁剪和调整PPT区域。
6. **生成PDF/PPT**：将处理后的PPT图片合并为一个文件。

## 弯路小知识

在目标检测中， Faster R-CNN 等模型默认输出的是矩形边界框。
如果需要检测不规则的四边形，可以使用 实例分割模型，比如 Mask R-CNN，它可以生成像素级的分割掩码，从而捕获不规则的形状。
