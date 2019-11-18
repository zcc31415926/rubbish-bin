#-*- coding:utf-8 -*-


from tgrocery import Grocery


train_src = [
    ('course', u'模拟电子技术基础'),
    ('course', u'大学数学微积分'),
    ('course', u'大学数学数学分析'),
    ('course', u'复变函数'),
    ('course', u'概率论与数理统计'),
    ('professional', u'单片机高级教程——应用与设计'),
    ('professional', u'精通正则表达式'),
    ('professional', u'Shell脚本学习指南'),
    ('professional', u'学习OpenCV'),
    ('professional', u'Vim实用技巧'),
    ('professional', u'Effective Python 编写高质量Python代码的59个有效方法'),
    ('course', u'线性代数'),
    ('professional', u'Visual Basic 程序设计实验指导与测试'),
    ('professional', u'汇编语言'),
    ('professional', u'编译原理'),
    ('course', u'数字基础'),
    ('professional', u'计算机视觉'),
    ('course', u'信号与系统'),
    ('course', u'数字信号处理教程'),
    ('professional', u'操作系统教程'),
    ('course', u'图论与代数结构'),
    ('course', u'C++程序设计题解与拓展'),
    ('course', u'C++程序设计实验指导'),
    ('english', u'英语同义词近义词例解词典'),
    ('english', u'TOEFL词汇'),
    ('extra', u'战争与和平'),
    ('extra', u'时间简史'),
    ('extra', u'微观经济学'),
    ('extra', u'宇宙的能级时空原理探析'),
    ('extra', u'岛上书店'),
    ('extra', u'资本论'),
    ('extra', u'Othello'),
    ('extra', u'第一哲学沉思集'),
    ('extra', u'瓦尔登湖'),
    ('extra', u'欲望心理学'),
    ('extra', u'魔鬼逻辑学'),
    ('extra', u'管理基础'),
    ('extra', u'理想国'),
    ('english', u'新概念英语4'),
    ('english', u'GRE考试官方指南'),
    ('english', u'新托福考试官方指南'),
    ('extra', u'牛奶可乐经济学'),
    ('course', u'自动控制理论与设计'),
    ('course', u'电力电子技术'),
    ('course', u'数字图像处理'),
    ('course', u'自动控制原理习题精解与考研指导'),
    ('course', u'现代检测技术'),
    ('extra', u'忒修斯之船'),
    ('professional', u'Arduino程序设计基础'),
    ('professional', u'机器学习导论'),
    ('professional', u'TensorFlow实战'),
    ('professional', u'Effective Modern C++'),
    ('extra', u'重新发现社会'),
    ('extra', u'Letter from an Unknown Woman')
]

# create a model named 'book_class'
grocery = Grocery('book_class')
grocery.train(train_src)
grocery.save()

# load the model 'book_class'
new_grocery = Grocery('book_class')
new_grocery.load()

# make predictions
str = raw_input('bookname:  ')
while str.strip():
    print('category: ', new_grocery.predict(str))
    str = raw_input('bookname:  ')
