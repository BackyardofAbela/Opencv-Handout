# OpenCV简介

**学习目标**

- 了解OpenCV是什么

- 能够独立安装OpenCV

- 了解OpenCV有哪些模块，熟悉OpenCV的基本数据结构


# 1 什么是OpenCV

## 1.1 OpenCV简介

![Snipaste_2019-09-23_16-41-04](assets/Snipaste_2019-09-23_16-41-04.png)

OpenCV是一款由Intel公司俄罗斯团队发起并参与和维护的一个计算机视觉处理开源软件库，支持与计算机视觉和机器学习相关的众多算法，并且正在日益扩展。

OpenCV的优势：

1. 编程语言

   OpenCV基于C++实现，同时提供python, Ruby, Matlab等语言的接口。OpenCV-Python是OpenCV的Python API，结合了OpenCV C++ API和Python语言的最佳特性。

2. 跨平台

   可以在不同的系统平台上使用，包括Windows，Linux，OS X，Android和iOS。基于CUDA和OpenCL的高速GPU操作接口也在积极开发中

3. 活跃的开发团队

4. 丰富的API

   完善的传统计算机视觉算法，涵盖主流的机器学习算法，同时添加了对深度学习的支持。

## 1.2 OpenCV-Python

OpenCV-Python是一个Python绑定库，旨在解决计算机视觉问题。

Python是一种由Guido van Rossum开发的通用编程语言，它很快就变得非常流行，主要是因为它的简单性和代码可读性。它使程序员能够用更少的代码行表达思想，而不会降低可读性。

与C / C++等语言相比，Python速度较慢。也就是说，Python可以使用C / C++轻松扩展，这使我们可以在C / C++中编写计算密集型代码，并创建可用作Python模块的Python包装器。这给我们带来了两个好处：首先，代码与原始C / C++代码一样快（因为它是在后台工作的实际C++代码），其次，在Python中编写代码比使用C / C++更容易。OpenCV-Python是原始OpenCV C++实现的Python包装器。

OpenCV-Python使用Numpy，这是一个高度优化的数据库操作库，具有MATLAB风格的语法。所有OpenCV数组结构都转换为Numpy数组。这也使得与使用Numpy的其他库（如SciPy和Matplotlib）集成更容易。

# 2 OpenCV部署方法

安装OpenCV之前需要先安装numpy, matplotlib。

创建Python虚拟环境cv, 在cv中安装即可。

先安装OpenCV-Python, 由于一些经典的算法被申请了版权，新版本有很大的限制，所以选用3.4.3以下的版本

```bash
pip install opencv-python==3.4.2.17
```

现在可以测试下是否安装成功，运行以下代码无报错则说明安装成功。

```python
import cv2
# 读一个图片并进行显示(图片路径需自己指定)
lena=cv2.imread("/Users/yaoxiaoying/Downloads/图片/1.jpg")
cv2.imshow("image",lena)
cv2.waitKey(0)
```

如果我们要利用SIFT和SURF等进行特征提取时，还需要安装：

```bash
pip install opencv-contrib-python==3.4.2.17
```



**总结**

1. OpenCV是计算机视觉的开源库

   优势：

   - 支持多种编程语言

   - 跨平台

   - 活跃的开发团队

   - 丰富的API

2. 能够独立的安装OpenCV-python

