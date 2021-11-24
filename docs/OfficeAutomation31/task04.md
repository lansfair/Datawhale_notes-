# Task 03 python与pdf
## 1.包的安装
**PyPDF2** :用于读取、写入、分割、合并PDF文件

**pdfplumber**:用于读取 PDF 文件中内容和提取 PDF 中的表格

## 2.pdf拆分
**拆分思路**:

    1.读取 PDF 的整体信息、总页数等
    2.遍历每一页内容，以每个 step 为间隔将 PDF 存成每一个小的文件块(也可以自定义间隔，如每5页保存一个pdf)
    3.将小的文件块重新保存为新的 PDF 文件
```python
import os
from PyPDF2 import PdfFileWriter, PdfFileReader
def split_pdf(filename, filepath, save_dirpath, step=5):
"""
拆分PDF为多个小的PDF文件，
@param filename:文件名
@param filepath:文件路径
@param save_dirpath:保存小的PDF的文件路径
@param step: 每step间隔的页面生成一个文件，例如step=5，表示0-4页、5-9页...为一个文件
@return:
"""
if not os.path.exists(save_dirpath):
os.mkdir(save_dirpath)
pdf_reader = PdfFileReader(filepath)
# 读取每一页的数据
pages = pdf_reader.getNumPages()
for page in range(0, pages, step):
    pdf_writer = PdfFileWriter()
    # 拆分pdf，每 step 页的拆分为一个文件
    for index in range(page, page+step):
    if index < pages:
    pdf_writer.addPage(pdf_reader.getPage(index))
    # 保存拆分后的小文件
    save_path = os.path.join(save_dirpath, filename+str(int(page/step)+1)+'.pdf')
    print(save_path)
    with open(save_path, "wb") as out:
    pdf_writer.write(out)
    print("文件已成功拆分，保存路径为："+save_dirpath)

filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
save_dirpath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报告
【拆分】')
split_pdf(filename, filepath, save_dirpath, step=5)
```
**注意**：由于编码问题需要修改PyPDF2中的`utils.py`，增加utf-8编码格式，将
```python
r = s.encode('latin-1')
if len(s) < 2:
bc[s] = r
return r
```
修改为
```python
try:
r = s.encode('latin-1')
if len(s) < 2:
bc[s] = r
return r
except Exception as e:
r = s.encode('utf-8')
if len(s) < 2:
bc[s] = r
return r
```
## 3.pdf合并
**合并思路**:

1.确定要合并的 文件顺序

2.循环追加到一个文件块中

3.保存成一个新的文件
```python
import os
from PyPDF2 import PdfFileReader, PdfFileWriter
def concat_pdf(filename, read_dirpath, save_filepath):
"""
合并多个PDF文件
@param filename:文件名
@param read_dirpath:要合并的PDF目录
@param save_filepath:合并后的PDF文件路径
@return:
"""
pdf_writer = PdfFileWriter()
# 对文件名进行排序
list_filename = os.listdir(read_dirpath)
list_filename.sort(key=lambda x: int(x[:-4].replace(filename, "")))
for filename in list_filename:
    print(filename)
    filepath = os.path.join(read_dirpath, filename)
    # 读取文件并获取文件的页数
    pdf_reader = PdfFileReader(filepath)
    pages = pdf_reader.getNumPages()
    # 逐页添加
    for page in range(pages):
        pdf_writer.addPage(pdf_reader.getPage(page))
    # 保存合并后的文件
    with open(save_filepath, "wb") as out:
        pdf_writer.write(out)
    print("文件已成功合并，保存路径为："+save_filepath)
filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
read_dirpath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报告
【拆分】')
save_filepath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报
告-合并后.pdf')
concat_pdf(filename, read_dirpath, save_filepath)
```
## 4.提取文字内容
```python
import os
import pdfplumber
def extract_text_info(filepath):
    """
    提取PDF中的文字
    @param filepath:文件路径
    @return:
    """
    with pdfplumber.open(filepath) as pdf:
        # 获取第2页数据
        page = pdf.pages[1]
        print(page.extract_text())

filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
# 提取文字内容
extract_text_info(filepath)
```
获取所有页文字
```python
with pdfplumber.open(filepath) as pdf:
    # 获取全部数据
    for page in pdf.pages
        print(page.extract_text())
```

## 5.提取表格内容
```python
import os
import pandas as pd
import pdfplumber

def extract_table_info(filepath):
    """
    提取PDF中的图表数据
    @param filepath:
    @return:
    """
    with pdfplumber.open(filepath) as pdf:
        # 获取第18页数据
        page = pdf.pages[17]
        # 如果一页有一个表格，设置表格的第一行为表头，其余为数据
        table_info = page.extract_table()
        df_table = pd.DataFrame(table_info[1:], columns=table_info[0])
        df_table.to_csv('dmeo.csv', index=False, encoding='gbk')

filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
# 提取表格内容
extract_table_info(filepath)
```

获取多个表格
```python
import os
import pandas as pd
import pdfplumber
def extract_table_info(filepath):
    """
    提取PDF中的图表数据
    @param filepath:
    @return:
    """
    with pdfplumber.open(filepath) as pdf:
        # 获取第7页数据
        page = pdf.pages[6]
        # 如果一页有多个表格，对应的数据是一个三维数组
        tables_info = page.extract_tables()
        for index in range(len(tables_info)):
            # 设置表格的第一行为表头，其余为数据
            df_table = pd.DataFrame(tables_info[index][1:],
            columns=tables_info[index][0])
        df_table.to_csv('dmeo.csv', index=False, encoding='gbk')
filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
# 提取表格内容
extract_table_info(filepath)
```
## 6.提取图片内容
将内容中的图片都提取出来，需要使用PyMuPDF中的fitz模块

**提取思路**:

1. 使用 fitz 打开文档，获取文档详细数据

2. 遍历每一个元素，通过正则找到图片的索引位置

3. 使用 Pixmap 将索引对应的元素生成图片

4. 通过 size 函数过滤较小的图片
```python
import os
import re
import fitz
def extract_pic_info(filepath, pic_dirpath):
    """
    提取PDF中的图片
    @param filepath:pdf文件路径
    @param pic_dirpath:要保存的图片目录路径
    @return:
    """
    if not os.path.exists(pic_dirpath):
        os.makedirs(pic_dirpath)
    # 使用正则表达式来查找图片
    check_XObject = r"/Type(?= */XObject)"
    check_Image = r"/Subtype(?= */Image)"
    img_count = 0

    """1. 打开pdf，打印相关信息"""
    pdf_info = fitz.open(filepath)
    # 1.16.8版本用法 xref_len = doc._getXrefLength()
    # 最新版本
    xref_len = pdf_info.xref_length()
    # 打印PDF的信息
    print("文件名：{}, 页数: {}, 对象: {}".format(filepath, len(pdf_info),
    xref_len-1))

    """2. 遍历PDF中的对象，遇到是图像才进行下一步，不然就continue"""
    for index in range(1, xref_len):
        # 1.16.8版本用法 text = doc._getXrefString(index)
        # 最新版本
        text = pdf_info.xref_object(index)
        is_XObject = re.search(check_XObject, text)
        is_Image = re.search(check_Image, text)
        # 如果不是对象也不是图片，则不操作
        if is_XObject or is_Image:
            img_count += 1
            # 根据索引生成图像
            pix = fitz.Pixmap(pdf_info, index)
            pic_filepath = os.path.join(pic_dirpath, 'img_' + str(img_count) +
            '.png')
            """pix.size 可以反映像素多少，简单的色素块该值较低，可以通过设置一个阈值过滤。以
            阈值 10000 为例过滤"""
            # if pix.size < 10000:
            # continue

            """3、将图像存为png格式"""
            if pix.n >= 5:
            # 先转换CMYK
            pix = fitz.Pixmap(fitz.csRGB, pix)
            # 存为PNG
            pix.writePNG(pic_filepath)

filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
pic_dirpath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报告
【文中图片】')
# 提取图片内容
extract_pic_info(filepath, pic_dirpath) 
```

## 7.转换为图片
将一页页的pdf转换为图片。需要使用pdf2image，还需要安装对应的组件poppler并添加环境变量
```python
import os
from pdf2image import convert_from_path, convert_from_bytes
def convert_to_pic(filepath, pic_dirpath):
    """
    每一页的PDF转换成图片
    @param filepath:pdf文件路径
    @param pic_dirpath:图片目录路径
    @return:
    """
    print(filepath)
    if not os.path.exists(pic_dirpath):
        os.makedirs(pic_dirpath)
    images = convert_from_bytes(open(filepath, 'rb').read())
    # images = convert_from_path(filepath, dpi=200)
    for image in images:
        # 保存图片
        pic_filepath = os.path.join(pic_dirpath,
'img_'+str(images.index(image))+'.png')
        image.save(pic_filepath, 'PNG')

# PDF转换为图片
convert_to_pic(filepath, pic_dirpath)

filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
pic_dirpath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报告
【转换为图片】')
# PDF转换为图片
convert_to_pic(filepath, pic_dirpath)
```

## 4.添加水印
主要思路是先生成一个水印pdf文件，再通过mergePage操作将pdf与水印pdf文件合并，得到带有水印的PDF文件。
添加水印代码：
```python
import os
from copy import copy
from PyPDF2 import PdfFileReader, PdfFileWriter

def add_watermark(filepath, save_filepath, watermark_filepath):
    """
    添加水印
    @param filepath:PDF文件路径
    @param save_filepath:最终的文件保存路径
    @param watermark_filepath:水印PDF文件路径
    @return:
    """
    """读取PDF水印文件"""
    # 可以先生成一个空白A4大小的png图片，通过https://mp.weixin.qq.com/s/_oJA6lbsdMlRRsBf6DPxsg 教程的方式给图片加水印，将图片插入到word中并最终生成一个水印PDF文档

    watermark = PdfFileReader(watermark_filepath)
    watermark_page = watermark.getPage(0)

    pdf_reader = PdfFileReader(filepath)
    pdf_writer = PdfFileWriter()
    for page_index in range(pdf_reader.getNumPages()):
        current_page = pdf_reader.getPage(page_index)
        # 封面页不添加水印
        if page_index == 0:
            new_page = current_page
        else:
            new_page = copy(watermark_page)
            new_page.mergePage(current_page)
        pdf_writer.addPage(new_page)
    # 保存水印后的文件
    with open(save_filepath, "wb") as out:
        pdf_writer.write(out)

filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
save_filepath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报
告-水印.pdf')
watermark_filepath = os.path.join(os.getcwd(), 'watermark.pdf')
# 添加水印
add_watermark(filepath, save_filepath, watermark_filepath)
```

## 5.文档加密与解密
貌似没有什么卵用

加密代码
```python
import os
from PyPDF2 import PdfFileReader, PdfFileWriter
def encrypt_pdf(filepath, save_filepath, passwd='xiaoyi'):
    """
    PDF文档加密
    @param filepath:PDF文件路径
    @param save_filepath:加密后的文件保存路径
    @param passwd:密码
    @return:
    """
    pdf_reader = PdfFileReader(filepath)
    pdf_writer = PdfFileWriter()

    for page_index in range(pdf_reader.getNumPages()):
        pdf_writer.addPage(pdf_reader.getPage(page_index))
    
    # 添加密码
    pdf_writer.encrypt(passwd)
    with open(save_filepath, "wb") as out:
        pdf_writer.write(out)

filename = '易方达中小盘混合型证券投资基金2020年中期报告.pdf'
filepath = os.path.join(os.getcwd(), filename)
save_filepath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报
告-加密后.pdf')
# 文档加密
encrypt_pdf(filepath, save_filepath, passwd='xiaoyi')
```

解密代码（通过密码解密）:
```python 
def decrypt_pdf(filepath, save_filepath, passwd='xiaoyi'):
    """
    解密 PDF 文档并且保存为未加密的 PDF
    @param filepath:PDF文件路径
    @param save_filepath:解密后的文件保存路径
    @param passwd:密码
    @return:
    """
    pdf_reader = PdfFileReader(filepath)
    # PDF文档解密
    pdf_reader.decrypt('xiaoyi')
    pdf_writer = PdfFileWriter()
    for page_index in range(pdf_reader.getNumPages()):
    pdf_writer.addPage(pdf_reader.getPage(page_index))
    with open(save_filepath, "wb") as out:
    pdf_writer.write(out)
    
filename = '易方达中小盘混合型证券投资基金2020年中期报告-加密后.pdf'
filepath = os.path.join(os.getcwd(), filename)
save_filepath = os.path.join(os.getcwd(), '易方达中小盘混合型证券投资基金2020年中期报
告-解密后.pdf')
# 文档解密
decrypt_pdf(filepath, save_filepath, passwd='xiaoyi')
```