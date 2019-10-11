# 直方图

**学习目标**

- 掌握图像的直方图计算和显示

- 了解掩膜的应用

- 熟悉直方图均衡化，了解自适应均衡化


# 1 灰度直方图



## 1.1 原理

直方图是对数据进行统计的一种方法，并且将统计值组织到一系列实现定义好的 bin 当中。其中， bin 为直方图中经常用到的一个概念，可以译为 “直条” 或 “组距”，其数值是从数据中计算出的特征统计量，这些数据可以是诸如梯度、方向、色彩或任何其他特征。

  图像直方图（Image Histogram）是用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素个数。这种直方图中，横坐标的左侧为较暗的区域，而右侧为较亮的区域。因此一张较暗图片的直方图中的数据多集中于左侧和中间部分，而整体明亮、只有少量阴影的图像则相反。

![image-20190928144352467](assets/image-20190928144352467.png)

注意：直方图是根据灰度图进行绘制的，而不是彩色图像。
  假设有一张图像的信息（灰度值 0 - 255，已知数字的范围包含 256 个值，于是可以按一定规律将这个范围分割成子区域（也就是 bins）。如：
$$
\left[0，255\right] = \left[0，15\right]\bigcup\left[16,30\right]\cdots\bigcup\left[240,255\right]
$$
  然后再统计每一个 bin(i) 的像素数目。可以得到下图（其中 x 轴表示 bin，y 轴表示各个 bin 中的像素个数）：
    ![image-20190928145730979](assets/image-20190928145730979.png)

直方图的一些**术语和细节**：

- dims：需要统计的特征数目。在上例中，dims = 1 ，因为仅仅统计了灰度值。
- bins：每个特征空间子区段的数目，可译为 “直条” 或 “组距”，在上例中， bins = 16。
- range：要统计特征的取值范围。在上例中，range = [0, 255]。

直方图的**意义**：

- 直方图是图像中像素强度分布的图形表达方式。   
- 它统计了每一个强度值所具有的像素个数。
- 不同的图像的直方图可能是相同的

## 1.2 直方图的计算和绘制

我们使用OpenCV中的方法统计直方图，并使用matplotlib将其绘制出来。

API：

```python
cv2.calcHist(images,channels,mask,histSize,ranges[,hist[,accumulate]])
```

参数：

- images: 原图像。当传入函数时应该用中括号 [] 括起来，例如：[img]。

- channels: 如果输入图像是灰度图，它的值就是 [0]；如果是彩色图像的话，传入的参数可以是 [0]，[1]，[2] 它们分别对应着通道 B，G，R。 　　

- mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如果你想统计图像某一部分的直方图的话，你就需要制作一个掩模图像，并使用它。（后边有例子） 　　

- histSize:BIN 的数目。也应该用中括号括起来，例如：[256]。 　　
- ranges: 像素值范围，通常为 [0，256]

示例：

如下图，绘制相应的直方图

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 直接以灰度图的方式读入
img = cv.imread('./image/cat.jpeg',0)
# 统计灰度图
histr = cv.calcHist([img],[0],None,[256],[0,256])
# 绘制灰度图
plt.figure(figsize=(10,6),dpi=100)
plt.plot(histr)
plt.grid()
plt.show()
```

![image-20190928155000064](assets/image-20190928155000064.png)

## 1.3 掩膜的应用

	上面我们使用cv.calcHist（）来查找完整图像的直方图。 如果要查找图像某些区域的直方图，该怎么办？ 只需在要查找直方图的区域上创建一个白色的蒙版图像，否则创建黑色， 然后将其作为掩码mask传递即可。

示例：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 直接以灰度图的方式读入
img = cv.imread('./image/cat.jpeg',0)
# 创建蒙版
mask = np.zeros(img.shape[:2], np.uint8)
mask[400:650, 200:500] = 255
# 掩模
masked_img = cv.bitwise_and(img,img,mask = mask)
# 统计掩膜后图像的灰度图
mask_histr = cv.calcHist([masked_img],[0],None,[256],[1,256])
histr = cv.calcHist([img],[0],None,[256],[0,256])

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img,cmap=plt.cm.gray)
axes[0,0].set_title("原图")
axes[0,1].imshow(mask,cmap=plt.cm.gray)
axes[0,1].set_title("蒙版数据")
axes[1,0].imshow(masked_img,cmap=plt.cm.gray)
axes[1,0].set_title("掩膜后数据")
axes[1,1].plot(mask_histr)
axes[1,1].grid()
axes[1,1].set_title("灰度直方图")
plt.show()
```

![image-20190928160241831](assets/image-20190928160241831.png)

# 2 直方图均衡化

## 2.1 原理与应用

	直方图均衡化是图像处理领域中增强图像对比度的一种方法。
	
	如果一副图像的像素占有很多的灰度级而且分布均匀，那么这样的图像往往有高对比度和多变的灰度色调。“直方图均衡化”是把原始图像的灰度直方图从比较集中的某个灰度区间变成在全部灰度范围内的均匀分布。直方图均衡化就是对图像进行非线性拉伸，重新分配图像像素值，使一定灰度范围内的像素数量大致相同。

![image-20190928162111755](assets/image-20190928162111755.png)

API：

```python 
dst = cv.equalizeHist(img)
```

参数：

- img: 灰度图像

返回：

- dst : 均衡化后的结果

示例：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 直接以灰度图的方式读入
img = cv.imread('./image/cat.jpeg',0)
# 均衡化处理
dst = cv.equalizeHist(img)

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img,cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(dst,cmap=plt.cm.gray)
axes[1].set_title("均衡化后结果")
plt.show()
```

![image-20190928163431354](assets/image-20190928163431354.png)

## 2.2 自适应的直方图均衡化

	上述的直方图均衡，我们考虑的是图像的全局对比度。 的确在进行完直方图均衡化之后，图片背景的对比度被改变了，在猫腿这里太暗，我们丢失了很多信息，所以在许多情况下，这样做的效果并不好。
	
	为了解决这个问题， 需要使用自适应的直方图均衡化。 此时， 整幅图像会被分成很多小块，这些小块被称为“tiles”（在 OpenCV 中 tiles 的 大小默认是 8x8），然后再对每一个小块分别进行直方图均衡化。 所以在每一个的区域中， 直方图会集中在某一个小的区域中）。如果有噪声的话，噪声会被放大。为了避免这种情况的出现要使用对比度限制。对于每个小块来说，如果直方图中的 bin 超过对比度的上限的话，就把 其中的像素点均匀分散到其他 bins 中，然后在进行直方图均衡化。最后，为了 去除每一个小块之间的边界，再使用双线性差值，对 每一小块进行拼接。

API：

```python
cv.createCLAHE(clipLimit, tileGridSize)
```

参数：

- clipLimit: 对比度限制，默认是40
- tileGridSize: 分块的大小，默认为$$8*8$$ 

示例：

```python
import numpy as np
import cv2 as cv
img = cv.imread('./image/cat.jpeg',0)
# 创建一个自适应均衡化的对象，并应用于原始图像
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img,cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(cl1,cmap=plt.cm.gray)
axes[1].set_title("自适应均衡化后的结果")
plt.show()
```

![image-20190928165605432](assets/image-20190928165605432.png)



**总结**

1. 灰度直方图：

   - 直方图是图像中像素强度分布的图形表达方式。 

   - 它统计了每一个强度值所具有的像素个数。
   - 不同的图像的直方图可能是相同的

   cv.calcHist（images，channels，mask，histSize，ranges [，hist [，accumulate]]）

2. 掩膜

   创建蒙版，透过mask进行传递，可获取感兴趣区域的直方图

3. 直方图均衡化：增强图像对比度的一种方法

   cv.equalizeHist(): 输入是灰度图像，输出是直方图均衡图像

4. 自适应的直方图均衡

   将整幅图像分成很多小块，然后再对每一个小块分别进行直方图均衡化，最后进行拼接

   clahe = cv.createCLAHE(clipLimit, tileGridSize)

