# 闲聊	
依稀记得几个月前和陈老哥吹牛说要分分钟搞定学校数据库自动化登录，结果稍微分析一下网页发现验证码无法以右键另存为的形式取到本地，貌似学校的验证码点击刷新的识别还是比较**sensitive**的。所以最近看了一些爬虫、机器学习的材料之后，上周末想要再尝试一下能不能做到识别验证码的一个效果，今天则是稍微作一个总结。

在开始之前，还是想说明一下，本文中的分析思路/基础代码框架主要参考了两个来源：[https://cuiqingcai.com/5052.html](https://cuiqingcai.com/5052.html)， 崔庆才老师的个人网点，也是我爬虫这一块一直仰慕的男神。 另外机器学习部分参考的是 **[澳]Robert Layton**的《Python数据挖掘入门与实践》的第八章。代码整体框架则是借鉴了崔老师在CookiesPool中底层代码中的一部分。[https://github.com/Python3WebSpider/CookiesPool](https://github.com/Python3WebSpider/CookiesPool)。

 # 环境准备
我非常喜欢用Anaconda下的Jupyter（初学者的最大特点），但是最近在做很多事情的时候发觉pycharm好像更适合一些偏探索/开发类的工作。不过要找到一个专业版也不容易哈。推荐使用上述两个环境或者Spyder，因为后续还有比较多的库引入，单独的Python使用起来稍微有一些复杂。

#  基本框架

![add image](https://github.com/siuszy/ecnu_login/raw/master/flowchart.png)


这里有两个地方需要稍作说明:

1 样本库是一个以验证码的实际值命名的图片集合，格式均为png，保存在本地，用于后续验证码识别中的匹配方法。

2 通过训练神经网络模型，若预测成功，将先前保存于本地的验证码名字更改为其实际值，扩充样本库。

# 代码样例
## 所依赖库的引入



```python
import os 
import time
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import numpy as np
from PIL import ImageDraw, ImageFont
from skimage import transform as tf
from skimage.measure import label, regionprops
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer 
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

import warnings
warnings.filterwarnings('ignore')
```

前半部分主要是一个**Selenium**的库，用于模拟浏览器的登录，这里使用的是Chrome，需要安装相应版本的driver; 后半部分则是**sklearn/pybrain**这两个库，主要是用于数据预处理和神经网络的搭建。

如果pip安装不了，就下载whl文件本地安装，应该不会有太大困难。最后则是一个防止warning输出的操作，为了美化输出，本意是因为担心自己熟悉的sklearn下的操作已经迁移到其他模块下导致warning输出；后续在代码中已经做了一定程度上的修改，唯一尚未解决的是skimage下的一个default模式改变的提示。

 # 初始化一个类

```python
class ecnu_login():
    def __init__(self, username, password, browser, zoom, random_seed=14, max_shear=0.2, modelname="testNet.xml",
                 train_model = True, fscore= False, report= False, templatefolder="D://python(siu)/template/"):
        """
        初始化浏览器及相应参数设置
        定义后续机器学习中模型所需的超参数
        :return: 
        """
        self.url = 'https://portal1.ecnu.edu.cn/cas/login?'
        self.browser = browser
        self.wait = WebDriverWait(self.browser, 20)
        self.username = username
        self.password = password
        self.zoom = zoom  #显示屏缩放比例 用于修改验证码位置
        
        self.TEMPLATES_FOLDER = templatefolder
        
        self.random_state = check_random_state(random_seed)
        self.numbers = list("0123456789")
        self.shear_values = np.arange(0, max_shear, 0.01)
        self.modelname = modelname
        self.signal = train_model # 判断是否要训练模型
        self.fscore = fscore # 是否输出测试集上预测结果
        self.report = report # 是否输出测试集上测试报告
```
一些很基础的操作，注释见上。唯一想讲一讲的是这个**zoom**参数，起初没注意屏幕缩放比例这一问题，导致一直截取不到验证码，后续要用这个参数对验证码的位置做一个缩放。

## 模拟登录的初步操作

```python
    def open(self):
        """
        打开网页输入用户名/密码并点击
        :return: None
        """
        print("打开公共数据库...")
        self.browser.get(self.url)
        username = self.wait.until(EC.presence_of_element_located((By.ID, 'un')))
        password = self.wait.until(EC.presence_of_element_located((By.ID, 'pd')))
        submit = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'login_box_landing_btn')))
        username.send_keys(self.username)
        password.send_keys(self.password)
        submit.click()
    
    def re_login(self):
        """
        输入用户名/密码/验证码并点击
        :return: None
        """
        username = self.wait.until(EC.presence_of_element_located((By.ID, 'un')))
        password = self.wait.until(EC.presence_of_element_located((By.ID, 'pd')))
        idecode = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'ide_code_input')))
        submit = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'login_box_landing_btn')))
        username.send_keys(self.username)
        password.send_keys(self.password)
        idecode.send_keys(self.idecode)
        submit.click()
        
    def password_error(self):
        """
        判断是否用户名或密码错误
        :return:
        """
        try:
            return WebDriverWait(self.browser, 3).until(
                EC.text_to_be_present_in_element((By.ID, 'errormsg'), '用户名密码错误'))
        except TimeoutException:
            return False
        
    def idecode_error(self):
        """
        判断是否验证码错误
        :return:
        """
        try:
            return WebDriverWait(self.browser, 3).until(
                EC.text_to_be_present_in_element((By.ID, 'errormsg'), '验证码有误'))
        except TimeoutException:
            return False
        
    def login_successfully(self):
        """
        判断是否登录成功
        :return:
        """
        try:
            return bool(
                WebDriverWait(self.browser, 5).until(EC.presence_of_element_located((By.CLASS_NAME,'PContentLeft'))))
        except TimeoutException:
            return False
```
这里也是一些比较基础的操作，其中登录这边用了显式等待的方法，防止网络连接问题，毕竟某normal大学的数据库，额...这个咱就不说了。不过也插一句题外话，某normal大学的校园网在"科学"上网不能够实现时，可能是一个下载python扩展库乃至于其他一些国外网站上软件的利器。
回到代码这边，open函数将被首先调用，re_login函数将在识别验证码后继续调用，比较类似。两个error函数是为了捕捉登陆时的错误，而最后一个login_successfully(）的函数则是通过返回一个布尔值去判断是否登录成功，函数内等待加载的的目标元素我选择了数据库旁的日历，如果有更新，那就做相应的更改。

##  验证码的获取

```python
    def get_position(self):
        """
        获取验证码位置
        :return: 验证码位置元组
        """
        try:
            img = self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'ide_code_img')))
        except TimeoutException:
            print('未出现验证码')
            self.open() # 回调open函数
        time.sleep(2)
        location = img.location
        size = img.size
        top, bottom, left, right = \
            location['y'], location['y'] + size['height'], location['x'], location['x'] + size['width']
        return (top, bottom, left, right)
    
    def get_screenshot(self):
        """
        获取网页截图
        :return: 截图对象
        """
        screenshot = self.browser.get_screenshot_as_png()
        screenshot = Image.open(BytesIO(screenshot))
        return screenshot
    
    def get_image(self, name='captcha.png'):
        """
        获取验证码图片
        :return: 图片对象
        """
        top, bottom, left, right = map(lambda x: x*self.zoom, self.get_position())
        # print('验证码位置', top, bottom, left, right)
        screenshot = self.get_screenshot()
        captcha = screenshot.crop((left, top, right, bottom))
        if not os.path.exists(self.TEMPLATES_FOLDER):
            os.makedirs(self.TEMPLATES_FOLDER) 
        captcha.save(self.TEMPLATES_FOLDER + "tmp.png") # 保存验证码至样本库
        return captcha
```
通过**Selenium**这个库提供的截图操作获取图片对象，然后用zoom参数调整位置参数并截取验证码，同时如果获取验证码失败就回调open()函数重新尝试打开数据库。然后将验证码暂时保存在一个文件夹内，在初始化类的时候我传入的路径是jupyter的工作目录，一般来讲可以用os.getcwd()获取，这里做了简化，直接传入。一系列操作之后，图片最终是以一个数组的形式去储存。

另外就是打开一个BytesIO()实现了在内存中读写bytes，这里显得比较鸡肋，因为已经做了本地保存。但是，在一些其他情形下，如果需要同时打开多个页面实现抓取验证码时，保存在本地就未必是一个合适的操作了。

## 识别验证码
很激动人心的部分，但由于个人水平有限，用的也是一些比较简单的方法。具体思路参照前述框架实现。
### 样本库匹配

```python
    def is_pixel_equal(self, image1, image2, x, y):
        """
        判断两个像素是否相同
        :param image1: 图片1
        :param image2: 图片2
        :param x: 位置x
        :param y: 位置y
        :return: 像素是否相同
        """
        # 取两个图片的像素点
        pixel1 = image1.load()[x, y]
        pixel2 = image2.load()[x, y]
        threshold = 20
        if abs(pixel1[0] - pixel2[0]) < threshold and abs(pixel1[1] - pixel2[1]) < threshold and abs(
                pixel1[2] - pixel2[2]) < threshold:
            return True
        else:
            return False
    
    def same_image(self, image, template):
        """
        识别相似验证码
        :param image: 待识别验证码
        :param template: 模板
        :return:
        """
        # 相似度阈值
        threshold = 0.99
        count = 0
        for x in range(image.width):
            for y in range(image.height):
                # 判断像素是否相同
                if self.is_pixel_equal(image, template, x, y):
                    count += 1
        result = float(count) / (image.width * image.height)
        if result > threshold:
            print('成功匹配')
            return True
        return False
    
    def detect_image(self, image):
        """
        匹配图片
        :param image: 图片
        :return: 如果匹配成功，返回匹配结果，否则返回None
        """
        print("正在遍历样本库匹配验证码...")
        for template_name in os.listdir(self.TEMPLATES_FOLDER):
            if template_name == "tmp.png": # 跳过缓存的验证码
                break
            # print('正在匹配', template_name)
            template = Image.open(self.TEMPLATES_FOLDER + template_name)
            if self.same_image(image, template):
                return template_name.split(".")[0]
                break
        return "样本库穷尽"
```
匹配主函数detect_image()接收一个新的验证码，然后遍历样本库内所有验证码图片去匹配，如果匹配成功就返回所得的验证码匹配结果，失败的话就返回一个提示性字符串。函数内调用了same_image()判断函数，而这个函数又调用了is_pixel_equal()函数。

主要思路是在R/G/B三个维度匹配给定位置两张图片像素点值是否相同，判断标准依赖于一个阈值；遍历所有像素点，基于相同像素点占总像素点的比例来判断获取的验证码是否已经在样本库中。

### 简单的神经网络识别
#### 工具函数
```python
    def create_captcha(self, number, shear=0, size=(36,36)):
        """
        :param image: 图片
        :param shear: 错切比例
        :return: 压缩后的像素数组
        """
        im = Image.new("L", size, "white") 
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(r"simhei.ttf", 44)
        draw.text((8,-5), number, fill=0, font=font)
        image = np.array(im)
        affine_tf = tf.AffineTransform(shear=shear,rotation=-0.05)
        image = tf.warp(image, affine_tf)
        return image/image.max()
        
    def segment_image(self,image):
        """
        :param image: 图片
        :return: 一个子图组
        """
        labeled_image = label(image < 1) # label查找像素相同又连接在一起的像素块
        subimages = []
        for region in regionprops(labeled_image):
            start_x, start_y, end_x, end_y = region.bbox
            subimages.append(image[start_x:end_x, start_y:end_y])
        if len(subimages) == 0:
            return [image,]
        return subimages 
```
第一个函数创建一个**灰度模式**下的数字对象用于后续生成训练集中单个的实例，创建方式中包含shear参数的tf.AffineTransform()是为了将一个常规生成的数字对象进行错切变换，通俗的讲，就是以仿射变换达到增加训练集多样性的目的。

第二个则是一个切割函数，寻找图像中连续的非白色像素块，基于连通块的思路去分割这个图象，并返回子图集合。

#### 生成训练图片集
```python
    def generate_sample(self):
        """
        随机生成图例及相应标签的方法
        :return: 一个图片集合及其对应标签
        """
        number = self.random_state.choice(self.numbers)
        shear = self.random_state.choice(self.shear_values)
        return self.create_captcha(number, shear=shear, size=(36, 36)) , self.numbers.index(number)
        
    def gen_dataset(self):
        """
        调用generate_sample 生成模型训练所需数据集
        :return: 特征数组/标签 以及相应划分
        """
        dataset, targets = zip(*(self.generate_sample() for i in range(3000))) # zip逆过程
        dataset = np.array(dataset, dtype='float')
        targets = np.array(targets)

        onehot = OneHotEncoder() # 目标模型最后输出10个值 ，选取最接近于1的值对应数字作为预测结果
        y = onehot.fit_transform(targets.reshape(targets.shape[0],1)) 
        y = y.todense() # 模块不支持稀疏矩阵

        # 转换为标准的18*18的图像
        dataset = np.array([resize(self.segment_image(sample,)[0], (18, 18), mode= 'constant') for sample in dataset]) 
        # 三维数组压缩成两维数组 一个维度代表数据集数量 一个维度包括了描述图像的数组
        X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2])) 

        # 划分训练集、测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)

        return X, y, X_train, X_test, y_train, y_test
```
这里第一个函数调用工具函数生成3000个数字图片训练集(设置随机种子的是为了方便结果复现）。

第二个函数接收第一个函数生成训练集特征数组和标签，并分别做了一定变化。

对于特征数据集合，调用工具函数首先截取有效区域，这时候图片的尺寸可能发生变化（已经不是36\*36了）；所以做了第二步压缩的操作，18\*18的规格接近于在实际中抓取验证码并切割后得到的尺寸；最后是将每个图片的18\*18个特征值矩阵压缩到一个维度上，最终特征值数据集的shape就从3000\*36\*36变为3000\*324。

对于标签集合，将其转化为0/1的形式，每个标签对应一个1\*10的向量，比如数字7就是在第七个位置为1，其余9个位置上为0。这样处理基于的思路是最后让模型输出10个值，如果结果像某个数字，使用近似于1的值；如果不像，就用近似于0的值。最后，以9:1的比例划分训练集和测试集，其实这里的测试集更像一个验证集。

#### 模型训练与预测
```python
    def train_model(self):
        """
        训练模型并保存
        :return: 
        """
        X, y, X_train, X_test, y_train, y_test = self.gen_dataset()

        training = SupervisedDataSet(X.shape[1], y.shape[1]) # 转化训练集形式
        for i in range(X_train.shape[0]):
            training.addSample(X_train[i], y_train[i])

        testing = SupervisedDataSet(X.shape[1], y.shape[1]) # 转化测试集形式
        for i in range(X_test.shape[0]):
            testing.addSample(X_test[i], y_test[i])

        net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True) # 传入网络各层的神经元个数

        trainer = BackpropTrainer(net, training, learningrate=0.01, weightdecay=0.01)  #设定学习率/误差函数偏导数

        trainer.trainEpochs(epochs=20) # 迭代20步

        if self.fscore:
            predictions = trainer.testOnClassData(dataset=testing) # 在实例上调用
            print("F-score: {0:.2f}".format(f1_score(y_test.argmax(axis=1).flatten().tolist()[0],
                                             predictions, average="micro")))

        if self.report:
            print(classification_report(y_test.argmax(axis=1), predictions))

        NetworkWriter.writeToFile(net,self.modelname) # 保存模型

    def predict_captcha(self, captcha_image):
        """
        调用训练好的模型并预测
        :param captcha_image: 图片
        :return: 模型识别出的验证码
        """
        net = NetworkReader.readFrom(self.modelname)
        subimages = self.segment_image(captcha_image)
        predicted_numbers = ""
        for i in range(2,len(subimages)-2): # 剔除截图时四个角
            subimage = resize(subimages[i], (18, 18), mode= 'constant')
            outputs = net.activate(subimage.flatten()) #输入324个值
            prediction = np.argmax(outputs) # 返回10个 找到最大的那个
            predicted_numbers += self.numbers[prediction]
        print("识别验证码为%s"%(predicted_numbers))
        return predicted_numbers
```
第一个函数中，模型的搭建与训练遵循**PyBrain**库中的一些很基本的操作，参数基本均保持为default值，最后将模型训练结果保存到本地。稍微值得一说的是，这里中间层的神经元个数我设成了100（没有做太多测试），所以三层神经元个数分别为324、100、10，比较符合经验上的“漏斗形”。另外就是在初始化类的过程中，定义了fscore和report两者bool值均为False，如果做相应修改可以输出模型训练后在测试集上的fscore和一个统计准确率、召回率等指标的分类文本报告。

第二个函数中，调用工具函数切割接收到的新验证码图片，并以一个简单的循环对子图集中的每一个子图做预测，并返回预测结果。

# 测试

```python
    def main(self):
        """
        登录
        :return:
        """
        
        self.open()
        
        # 如果不需要验证码直接登录成功
        if self.login_successfully():
            '''
            仅作测试用，可以在open函数的submit.click()前加入time.sleep()等待，用于手动输入验证码
            '''
            print("登陆成功（手动输入验证码）")
        
        number_try = 1
        
        while self.login_successfully() is not True:
            
            time.sleep(2)
            print("第%d次尝试登录公共数据库..."%(number_try))
            number_try += 1
            
            # 获取验证码图片
            image = self.get_image('captcha.png')
            img = np.array(image.convert(mode= 'L')) # 灰度模式打开
            img = img/img.max() # 压缩像素值
            
            if self.detect_image(image) == "样本库穷尽":
                print("样本库穷尽")
                # 识别验证码
                if self.signal:
                    print("正在训练模型...")
                    self.train_model() # 训练识别模型并保存
                    print("模型训练完成")
                    self.signal = False # 不再继续训练模型
                    self.idecode = self.predict_captcha(img) # 调用模型并识别
                else:
                    self.idecode = self.predict_captcha(img) # 调用模型并识别  
                # 重新登录
                self.re_login()
            else:
                self.idecode = self.detect_image(image)
                # 重新登录
                self.re_login()
                
            # 错误捕捉
            if self.password_error():
                print('用户名或密码错误,登录失败...')
                os.remove(self.TEMPLATES_FOLDER + "tmp.png") # 谨慎考虑仍然删去本次所获得验证码
            elif self.idecode_error():
                print('验证码有误,登陆失败...')
                os.remove(self.TEMPLATES_FOLDER + "tmp.png")
            else:
                pass
        #
        print("登录成功")
        # 识别成功验证码入库
        try:
            os.rename(self.TEMPLATES_FOLDER + "tmp.png", self.TEMPLATES_FOLDER + self.idecode + ".png")
        except:
            pass

browser = webdriver.Chrome()
if __name__ == '__main__':
    ecnu_login(username = 'Matthew Healy',  # 用户名
               password = 'the1975', # 密码
               browser = browser, 
               zoom = 1.25).main()
```
这里通过一个main()函数封装了所有需要回调的函数，遵循的就是前述的框架，并在测试过程中打印一些提示性内容。其中get_image()函数获取验证码图片对象后改用灰度模式重新打开，该模式与之前提到的create_captcha()工具函数生成的对象保持一致。
之后就是愉快的测试效果的部分了。

![add image](https://github.com/siuszy/ecnu_login/raw/master/test.png)

这里我没有调用本地保存的模型，进行了重新训练并测试，挑选了尝试次数比较少的一个结果，易于展示。当然，别的一些测试中，一般来讲至多10次尝试之后能够准确识别出验证码。换个角度，用不严谨的一个数学思维来说，如果验证码能够比较成功的识别10个数字中的6个，单次尝试成功概率大约是13%，主函数封装后包含一个回调机制（通俗点，死循环），所以理论上尝试次数不会太多。

这里插一句题外话，如果有了解jupyter-themes插件的朋友，耐心阅读到这里的你请不要觉得这张测试结果样例背景色非常丑，因为我身边所有的好友都已经吐槽过了。但在这里，还是强烈推荐，**gruvboxl**，护眼主题色，你值得拥有！

# 写在最后
反思一下代码的不足之处，这是非常有必要的；样本外测试结果（也就是实际登录数据库的过程）也说明准确率还是有待提高。我这里仅罗列以下几点，主要关注于机器学习部分：
| 模块           | 可优化部分                                                   |
| -------------- | ------------------------------------------------------------ |
| 工具函数       | 训练集图片生成方式（大小、字体、字号、像素、错切值等），切割方法（像素值连块判断的逻辑） |
| 训练集构建     | 样本数量，特征值、标签值处理方法                             |
| 模型超参数选择 | 神经网络中间层神经元个数（预设为100）乃至于层数，输出值的个数，模型训练中的学习率（0.01），误差偏导数（0.01），迭代步数（预设为20） |
| 测试部分       | 多次测试可以发现在哪些数字上容易混淆，进一步针对性做优化     |

写到这里发现：**嗯?? 好像都可以优化。**  不过想来也是无聊之举，仅供自己和大家娱乐一下。
