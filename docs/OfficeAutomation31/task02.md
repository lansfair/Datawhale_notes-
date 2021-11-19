# Task02 Python自动化之Excel
## 1.包的安装

xlsx格式：open

xls格式： xlwt 模块写，xlrd 模块读取

## 2.Excel读取
1. 打开excel：openpyxl.load_workbook('XXX.xlsx')
2. 表名、类型、大小的读取
3. 读取单元格：与pandas类似，value属性，包含这个单元格中保存的值，有row、column和coordinate属性，提供该单元格的位置信息

4.练习题：

找出用户行为偏好.xlsx中sheet1表中空着的格子，并输出这些格子的坐标
```python
from openpyxl import load_workbook
exl = load_workbook('用户行为偏好.xlsx')
sheet = exl.active
for row in sheet.iter_rows(min_row = 1, max_row = 29972,
min_col = 1, max_col = 10):
#具体查看对应表格的行列数
for cell in row:
if not cell.value:
print(cell.coordinate)
```

## 3.Excel写入
1.写入数据并保存：

    1). 原有工作簿中写入数据并保存:exl.save(filename = 'xxx.xlsx')
    2). 创建新的表格写入数据并保存：创建表，创建sheet，写入excel
    3). 也可将公式写入单元格保存

2. 插入数据：

    1). 插入列数据：sheet.insert_cols(idx=2, amount=5)
    
    2). 插入行数据：sheet.insert_rows(idx=2, amount=5)

3. 删除数据：

    1). 删除多列：sheet.delete_cols(idx=5, amount=2)

    2). 删除多行: sheet.delete_rows(idx=2, amount=5)
4. 移动：
    
    当数字为正即向下或向右，为负即为向上或向左

        sheet.move_range('B3:E16',rows=1,cols=-1)

5. sheet表操作：

    1)创建sheet：
            
            exl.create_sheet('new_sheet')

    2)修改表名：

        sheet = exl.active
        sheet.title = 'newname'
    
## 4.Excel 样式操作
1. 设置字体样式

    Font(name字体名称,size大小,bold粗体,italic斜体,color颜色)

2. 设置对齐样式

    水平对齐： distributed, justify, center, left, fill, centerContinuous, right, general

    垂直对齐： bottom, distributed, justify, center, top

3. 设置边缘框样式

    Side ：变现样式，边线颜色等
    Border ：左右上下边线
    变现样式： double, mediumDashDotDot, slantDashDot, dashDotDot, dotted, hair,
    mediumDashed, dashed, dashDot, thin, mediumDashDot, medium, thick

4. 设置行高与列宽

    sheet.row_dimensions[1].height = 50
    sheet.column_dimensions['C'].width = 20 

5.合并、取消合并单元格

    sheet.merge_cells(start_row=1, start_column=3,
    end_row=2, end_column=4)
    sheet.unmerge_cells(start_row=1, start_column=3,
    end_row=2, end_column=4)

6.练习题：

打开test文件，找出文件中购买数量buy_mount 超过5的行，并对其标红、加粗、附上边框。

没有找到这个test文件

```python
from openpyxl import load_workbook
from openpyxl.styles import Font, Side, Border
workbook = load_workbook('./test.xlsx')
sheet = workbook.active
buy_mount = sheet['F']
row_lst = []
for cell in buy_mount:
    if isinstance(cell.value, int) and cell.value > 5:
        print(cell.row)
        row_lst.append(cell.row)
side = Side(style='thin', color='FF000000')
border = Border(left=side, right=side, top=side, bottom=side)
font = Font(bold=True, color='FF0000')
for row in row_lst:
    for cell in sheet[row]:
        cell.font = font
        cell.border = border
workbook.save('new_test.xlsx')
```