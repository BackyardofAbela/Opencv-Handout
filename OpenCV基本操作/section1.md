# 算数操作

**学习目标**

- 了解图像的加法、混合和按位操作

# 1.图像的加法

你可以使用OpenCV的cv.add()函数把两幅图像相加，或者可以简单地通过numpy操作添加两个图像，如res = img1 + img2。两个图像应该具有相同的大小和类型，或者第二个图像可以是标量值。

**注意：OpenCV加法和Numpy加法之间存在差异。OpenCV的加法是饱和操作，而Numpy添加是模运算。**

参考以下代码：

```python
>>> x = np.uint8([250])
>>> y = np.uint8([10])
>>> print( cv.add(x,y) ) # 250+10 = 260 => 255
[[255]]
>>> print( x+y )          # 250+10 = 260 % 256 = 4
[4]
```

这种差别在你对两幅图像进行加法时会更加明显。OpenCV 的结果会更好一点。所以我们尽量使用 OpenCV 中的函数。

# 2.图像的混合

这其实也是加法，但是不同的是两幅图像的权重不同，这就会给人一种混合或者透明的感觉。图像混合的计算公式如下：

> g(x) = (1−α)f0(x) + αf1(x)

通过修改 α 的值（0 → 1），可以实现非常炫酷的混合。

现在我们把两幅图混合在一起。第一幅图的权重是0.7，第二幅图的权重是0.3。函数cv2.addWeighted()可以按下面的公式对图片进行混合操作。

> dst = α⋅img1 + β⋅img2 + γ

这里γ取为零。

参考以下代码：

```python
img1 = cv.imread('ml.png')
img2 = cv.imread('opencv-logo.png')
dst = cv.addWeighted(img1,0.7,img2,0.3,0)
plt.imshow(dst[:,:,::-1])
```

窗口将如下图显示：

![image3](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/3.Core%20Operations/Image/image3.png)

# 3.图像按位操作

这里包括的按位操作有：AND，OR，NOT，XOR 等。当我们提取图像的一部分，选择非矩形ROI时这些操作会很有用。下面的例子就是教给我们如何改变一幅图的特定区域。把OpenCV的标志放到另一幅图像上。如果使用图像的加法，图像的颜色会改变，如果使用图像的混合，会得到一个透明的效果，但是我不希望它透明。我们可以通过下面的按位运算实现：

```python
# 1 读取图像
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo-white.png')
# 2 创建ROI区域
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# 3 创建蒙版
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# 4 蒙版
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# 5 按位与添加logo.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()
```

窗口将如下图显示。左面的图像是我们创建的模板，右边的是最终结果。为了帮助大家理解，我把上面程序的中间结果也显示了出来，特别是img1_bg和img2_fg。

![image4](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/3.Core%20Operations/Image/image4.png)

**总结**

1. 图像加法：cv.add()

2. 图像的混合：cv.addweight()

3. 图像的位操作：

   与：AND

   或：OR

   非：NOT

   异或：XOR

