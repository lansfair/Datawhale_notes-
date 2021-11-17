# Task01 文件自动化与邮件处理
## 1. 文件处理
    1.路径的结合与拆分
    2.访问当前目录和改变目录
    3.绝对路径与相对路径及路径操作
    4.路径有效性检查
    5.文件夹及文件的读写，保存，查看大小
    6.复制、删除、移动、遍历文件夹和文件
    7.文件压缩与解压
## 2. 自动发送电子邮件
    smtplib和email，能够实现邮件功能。
    smtplib库负责发送邮件。
    email库负责构造邮件格式和内容。

## 3.练习
    1.如果已有的文件以写模式打开，会发生什么？
        答：如果写入内容，会覆盖原文件。

    2.read() 和readlines() 方法之间的区别是什么？
        答：read()方法的返回值是字符串类型
            readlines()方法的返回值是列表类型
    
    3.一、生成随机的测验试卷文件
    假如你是一位地理老师，班上有35名学生，你希望进行美国各州首府的一个小测验。不妙的是，班里有几个坏蛋，你无法确信学生不会作弊。你希望随机调整问题的次序，这样每份试卷都是独一无二的， 这让任何人都不能从其他人那里抄袭答案。当然，手工
    完成这件事又费时又无聊。好在，你懂一Python。
    下面是程序所做的事：
    • 创建 35 份不同的测验试卷。
    • 为每份试卷创建 50 个多重选择题，次序随机。
    • 为每个问题提供一个正确答案和 3 个随机的错误答案，次序随机。
    • 将测验试卷写到35个文本文件中。
    • 将答案写到35个文本文件中。
    这意味着代码需要做下面的事：
    • 将州和它们的首府保存在一个字典中。
    • 针对测验文本文件和答案文本文件，调用open()、 write()和close()。
    • 利用random.shuffle()随机调整问题和多重选项的次序。
```python
import random

# 以美国的洲名为键, 以洲的首府作为值
capitals = {
    'Alabama': 'Montgomery',
    'Alaska': 'Juneau',
    'Arizona': 'Phoenix',
    'Arkansas': 'Little Rock',
    'California': 'Sacramento',
    'Colorado': 'Denver',
    'Connecticut': 'Hartford',
    'Delaware': 'Dover',
    'Florida': 'Tallahassee',
    'Georgia': 'Atlanta',
    'Hawaii': 'Honolulu',
    'Idaho': 'Boise',
    'Illinois': 'Springfield',
    'Indiana': 'Indianapolis',
    'Iowa': 'Des Moines',
    'Kansas': 'Topeka',
    'Kentucky': 'Frankfort',
    'Louisiana': 'Baton Rouge',
    'Maine': 'Augusta',
    'Maryland': 'Annapolis',
    'Massachusetts': 'Boston',
    'Michigan': 'Lansing',
    'Minnesota': 'Saint Paul',
    'Mississippi': 'Jackson',
    'Missouri': 'Jefferson City',
    'Montana': 'Helena',
    'Nebraska': 'Lincoln',
    'Nevada': 'Carson City',
    'New Hampshire': 'Concord',
    'New Jersey': 'Trenton',
    'New Mexico': 'Santa Fe',
    'New York': 'Albany',
    'North Carolina': 'Raleigh',
    'North Dakota': 'Bismarck',
    'Ohio': 'Columbus',
    'Oklahoma': 'Oklahoma City',
    'Oregon': 'Salem',
    'Pennsylvania': 'Harrisburg',
    'Rhode Island': 'Providence',
    'South Carolina': 'Columbia',
    'South Dakota': 'Pierre',
    'Tennessee': 'Nashville',
    'Texas': 'Austin',
    'Utah': 'Salt Lake City',
    'Vermont': 'Montpelier',
    'Virginia': 'Richmond',
    'Washington': 'Olympia',
    'WestVirginia': 'Charleston',
    'Wisconsin': 'Madison',
    'Wyoming': 'Cheyenne'}

# 为 35 位同学生成试卷
for quizNum in range(35):
    # 打开一个试卷文件
    quizFile = open('./capitalsquiz%s.txt' % (quizNum + 1), 'w')
    # 打开一个答案文件
    answerKeyFile = open('./capitalsquiz_answers%s.txt' % (quizNum + 1), 'w')

    # 在卷子开头打印出一些信息
    quizFile.write('Name:\n\nDate:\n\nPeriod:\n\n')
    quizFile.write((' ' * 20) + 'State Capitals Quiz (Form %s)' % (quizNum + 1))
    quizFile.write('\n\n')

    # 打乱洲名
    states = list(capitals.keys())
    random.shuffle(states)

    # 每张试卷里面 50 道题目
    for questionNum in range(50):
        # 拿到正确的答案
        correctAnswer = capitals[states[questionNum]]

        # 生成排除正确答案之外的错误答案列表 错误的答案要从这里面随机生成
        wrongAnswers = list(capitals.values())
        del wrongAnswers[wrongAnswers.index(correctAnswer)]
        wrongAnswers = random.sample(wrongAnswers, 3)

        answerOptions = wrongAnswers + [correctAnswer]
        # 随机一下选择列表
        random.shuffle(answerOptions)

        # 写题目
        quizFile.write('%s. What is the capital of %s?\n' % (questionNum + 1, states[questionNum]))
        for i in range(4):
            quizFile.write(' %s. %s\n' %('ABCD'[i], answerOptions[i]))
            quizFile.write('\n')

        # 写入答案
        answerKeyFile.write('%s. %s\n' % (questionNum + 1, 'ABCD'[answerOptions.index(correctAnswer)]))

    quizFile.close()
    answerKeyFile.close()
```
    4.编写一个程序，遍历一个目录树，查找特定扩展名的文件（诸如.pdf 或.jpg）。不论这些文件的位置在哪里，将它们拷贝到一个新的文件夹中
```PYTHON
# 导入模块
import os
import shutil
# 给予遍历地址
path = r'.\OfficeAutomation'  
# 给予复制图片的位置
new_path = r'.\new_dir'  
# 用来判断图片的拓展名
list = ['jpg', 'png']  
for dirpath, dirlist, filelist in os.walk(path):  # 遍历目录
    for file in filelist:  # 遍历文件
        # 切割取后端用来对比是否是图片
        if file.split('.')[-1] in list:  
            图片地址 = dirpath + '/' + file  # 拼接图片地址
            shutil.copy(图片地址, new_path)  # 复制图片到新地址
print('复制完成')
```
5.一些不需要的、巨大的文件或文件夹占据了硬盘的空间，这并不少见。如果你试图释放计算机
上的空间，那么删除不想要的巨大文件效果最好。但首先你必须找到它们。编写一个程序，遍历一个目录树，查找特别大的文件或文件夹， 比方说，超过100MB的文件（回忆一下，要获得文件的大小，可以使用 os 模块的 os.path.getsize() ）。将这些文件的绝对路径打印到屏幕
```python
def move_size(file_path,maxsize=100):
	for file_p , _ , file in os.walk(file_path):
		for i in file:
			if os.path.getsize(os.path.join(file_p,i))>=100*1024*1024:
				# print(i)
				flag=input('{}是否删除(y/n):'.format(i))
				if flag=='y':
					os.unlink(os.path.join(file_path,i))
				else:
					pass
```
6.编写一个程序，在一个文件夹中，找到所有带指定前缀的文件，诸如 spam001.txt, spam002.txt 等，并定位缺失的编号（例如存在 spam001.txt 和 spam003.txt，但不存 在 spam002.txt）。让该程序对所有后面的文件改名，消除缺失的编号。
```python
import os,re,shutil#导入模块
path=input('请输入查找路径:')
prefixname=input('请输入前缀:')#如spam,不包括编号
rg=re.compile(r'%s\d\d\d.*' %prefixname)#创建匹配特定前缀的正则表达式
rg2=re.compile(r'\d\d\d')#创建匹配编号的正则表达式
L=[]#储存文件名
for filename in os.listdir(path):#找出所有特定前缀的文件，并存放在L里
    if os.path.isfile(os.path.join(path,filename)) and rg.search(filename):
        L.append(filename)
print(L)
for i in range(len(L)):#迭代L
    if i==0:
        continue
    if (int(rg2.search(L[i]).group())-int(rg2.search(L[i-1]).group()))>1:#判断是否有编号缺失
        num=int(rg2.search(L[i-1]).group())+1#修改后的文件编号
        exname=os.path.splitext(os.path.join(path,L[i]))[1]#文件后缀名
        filename=r'%s%03d%s' %(prefixname,num,exname)#新文件名
        shutil.move(os.path.join(path,L[i]),os.path.join(path,filename))#则重命名文件
        L[i] = filename#替换L中原有文件名，以便下一次比较
print(L)
```