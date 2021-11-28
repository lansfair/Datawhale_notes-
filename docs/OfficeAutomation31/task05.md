# task05 爬虫入门与综合应用
## 1.使用的库
1.**requests**:用于对网页进行爬取
2.**BeautifulSoup**：用于解析html页面，提取信息
## 2.requests的基本使用
**1.访问百度**
```python
import requests
# 发出http请求
re=requests.get("https://www.baidu.com")
# 查看响应状态
print(re.status_code)
#输出：200
#200就是响应的状态码，表示请求成功
#我们可以通过res.status_code的值来判断请求是否成功。
```
**HTTP响应状态码**参考：
- 1xx：信息
- 2xx：成功
- 3xx：重定向
- 4xx：客户端错误
- 5xx：服务器错误

**2.下载txt文件**
```python
import requests
# 发出http请求
re = requests.get('https://apiv3.shanbay.com/codetime/articles/mnvdu') # 查看响应状态
print('网页的状态码为%s'%re.status_code)
with open('鲁迅文章.txt', 'w') as file:
    # 将数据的字符串形式写入文件中  \
    print('正在爬取小说')  
    file.write(re.text)
```
**3.下载图片**
```python
import requests
# 发出http请求
#下载图片
res=requests.get('https://img-blog.csdnimg.cn/20210424184053989.PNG') # 以二进制写入的方式打开一个名为 info.jpg 的文件
with open('datawhale.png','wb') as ff:
    # 将数据的二进制形式写入文件中  print('爬取图片')
    ff.write(res.content)
```
总结：

- re.text用于文本内容的获取、下载
- re.content用于图片、视频、音频等内容的获取、下载
- re.encoding 爬取内容的编码形式，常见的编码方式有 ASCII、GBK、UTF-8 等。如果用和文件编码不同的方式去解码，我们就会得到一些乱码。
## 3.HTML解析和提取
**1.浏览器工作原理**

向浏览器中输入某个网址，浏览器回向服务器发出请求，然后服务器就会作出响应。其实，服务器返回给浏览器的这个结果就是HTML代码，浏览器会根据这个HTML代码将网页解析显示成正常的页面

**2.BeautifulSoup**

在使用爬虫时，我们也需要去对爬取到的HTML界面进行相应的解析，通常使用BeautifulSoup库来完成。例如解析豆瓣读书Top250：

首先将网页源代码的字符串形式解析为BeautifulSoup对象
```python
import io
import sys
import requests
from bs4 import BeautifulSoup
###运行出现乱码时可以修改编码方式
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
###
headers = {'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'}
res = requests.get('https://book.douban.com/top250', headers=headers) 
soup = BeautifulSoup(res.text, 'lxml')
print(soup)
```

使用find()和find_all()从BeautifulSoup对象中提取数据
- ﬁnd() 返回符合条件的首个数据
- ﬁnd_all() 返回符合条件的所有数据
```python
import io
import sys
import requests
from bs4 import BeautifulSoup
# 如果出现了乱码报错，可以修改编码形式
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
headers ={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'}
res = requests.get('https://book.douban.com/top250', headers=headers) 
soup = BeautifulSoup(res.text, 'lxml')
print(soup.find('a'))
# 返回一个列表 包含所有的<a>标签
print(soup.find_all('a'))
```

- 除了传入 HTML 标签名称外，BeautifulSoup 还支持熟悉的定位

```python
# 定位div开头 同时id为'doubanapp-tip的标签
soup.find('div', id='doubanapp-tip')
# 定位a抬头 同时class为rating_nums的标签
soup.find_all('span', class_='rating_nums')
#class是python中定义类的关键字，因此用class_表示HTML中的class
```
## 4.项目实践
**1.自如公寓数据抓取**
```python
# 导包
import requests
from bs4 import BeautifulSoup
import random
import time
import csv


# 找ua伪装，网上寻找的ua伪装
user_agent = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)"]

def get_info():
    csvheader=['名称','面积','朝向','户型','位置','楼层','是否有电梯','建成时间',' 门锁','绿化']
    with open('wuhan_ziru.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvheader)
        for i in range(1, 50):
            # 总共有50页
            print('正在爬取自如第%s页' % i)
            timelist = [1, 2, 3]
            print('有点累了，需要休息一下啦（￢㉨￢）')
            time.sleep(random.choice(timelist))

            url = 'https://wh.ziroom.com/z/p%s/' % i
            headers = {'User-Agent': random.choice(user_agent)}
            r = requests.get(url, headers=headers)
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, 'lxml')
            all_info = soup.find_all('div', class_='info-box')
            print('开始干活咯(๑>؂ ๑）')
            for info in all_info:
                href = info.find('a')
                if href != None:
                    href = 'https:' + href['href']
                    try:
                        print('正在爬取%s' % href)
                        house_info = get_house_info(href)
                        writer.writerow(house_info)
                    except:
                        print('出错啦，%s进不去啦( •̥́ ˍ •̀􀀁 )' % href)

def get_house_info(href):
    # 得到房屋的信息
    time.sleep(1)
    headers = {'User-Agent': random.choice(user_agent)}
    response = requests.get(url=href, headers=headers)
    response = response.content.decode('utf-8', 'ignore')
    soup = BeautifulSoup(response, 'lxml')
    name = soup.find('h1', class_='Z_name').text
    sinfo = soup.find('div', class_='Z_home_b clearfix').find_all('dd')
    area = sinfo[0].text
    orien = sinfo[1].text
    area_type = sinfo[2].text
    dinfo = soup.find('ul', class_='Z_home_o').find_all('li')
    location = dinfo[0].find('span', class_='va').text
    loucen = dinfo[1].find('span', class_='va').text
    dianti = dinfo[2].find('span', class_='va').text
    niandai = dinfo[3].find('span', class_='va').text
    mensuo = dinfo[4].find('span', class_='va').text
    lvhua = dinfo[5].find('span', class_='va').text
    room_info =['名称', '面积', '朝向', '户型', '位置', '楼层', '是否有电梯', '建成时间', ' 门锁', '绿化']
    return room_info

if __name__ == '__main__':
    get_info()
```
**2. 36kr信息抓取**
```python
import requests
import random
from bs4 import BeautifulSoup
import smtplib # 发送邮件模块
from email.mime.text import MIMEText # 定义邮件内容
from email.header import Header  # 定义邮件标题
smtpserver = 'smtp.qq.com' #设置服务器

# 发送邮箱用户名密码
user = '******@qq.com'
password = '*********'
# 发送和接收邮箱
sender = '******qq.com'
receive = '******@qq.com'

user_agent = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)"]

def send_email(content):
    # 通过QQ邮箱发送
    title = '36kr快讯'
    subject = title
    msg = MIMEText(content, 'html', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = sender
    msg['To'] = receive
    # SSL协议端口号要使用465
    smtp = smtplib.SMTP_SSL(smtpserver, 465)  # 这里是服务器端口！  # HELO 向服务器标识用户身份
    smtp.helo(smtpserver)
    # 服务器返回结果确认
    smtp.ehlo(smtpserver)
    # 登录邮箱服务器用户名和密码
    smtp.login(user, password)
    smtp.sendmail(sender, receive, msg.as_string())
    smtp.quit()

def main():
    print('正在爬取数据')
    url = 'https://36kr.com/newsflashes'
    headers = {'User-Agent': random.choice(user_agent)}
    response = requests.get(url,headers=headers)
    response = response.content.decode(
        'utf-8', 'ignore')
    soup = BeautifulSoup(response, 'lxml')
    news = soup.find_all('a', class_='item-title')
    news_list = []
    for i in news:
        title = i.get_text()
        href = 'https://36kr.com' + i['href']
        news_list.append(title + '<br>' + href)
    info = '<br></br>'.join(news_list)
    print('正在发送信息')
    send_email(info)

if __name__ == '__main__':
    main()
```

## 注意：在实际使用爬虫的过程中，使用高匿代理是十分有必要的