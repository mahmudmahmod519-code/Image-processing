from enum import Enum
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import pandas as pd
from glob import glob



class Image_process:
  class mode_color(Enum):
    RED=1

  class Mode(Enum):
    NEGATIVE_MODE=-1
    SHRINK_MODE=0
    # SLID_MODE=1
    BULR_MODE=1
    STRITCH_MODE=2
    DAREKER_MODE=3
    WHITER_MODE=4
    GRAY_MODE=5

    # mode_feature=cv2.IMREAD_ANYCOLOR
    #by_color=cv2.COLOR_RGB2GRAY

  def __init__(self,src):
    self.__src=cv2.imread(src)

  def image_color_mode(self):
    #transfare color
    matrix_image_color_mode=cv2.cvtColor(self.__src,cv2.COLOR_BayerGB2BGRA)

    cv2_imshow(matrix_image_color_mode)


  def image_filter_mode(self,mode):
    if(mode==self.Mode.NEGATIVE_MODE.value):
      self.__src=255-self.__src
      cv2_imshow(self.__src)

    elif(mode==self.Mode.SHRINK_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=.5,beta=0)
      cv2_imshow(self.__src)

    elif(mode==self.Mode.STRITCH_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=1.5,beta=0)
      cv2_imshow(self.__src)

    elif(mode==self.Mode.WHITER_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=1,beta=50)
      cv2_imshow(self.__src)

    elif(mode==self.Mode.DAREKER_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=1,beta=-50)
      cv2_imshow(self.__src)

    elif(mode==self.Mode.BULR_MODE.value):
      self.__src=cv2.blur(self.__src,(10,20))
      cv2_imshow(self.__src)

    elif(mode==self.Mode.GRAY_MODE.value):
      self.__src=cv2.cvtColor(self.__src,cv2.COLOR_RGB2GRAY)
      cv2_imshow(self.__src)
      # show original image
    # cv2_imshow(cv2.rotate(self.__src))
    # cv2_imshow(np.compress(self.__src))


    # blur=cv2.GaussianBlur(self.__src,(5,5),0)
    # cv2_imshow(blur)





  def draw_histogram(self):
    # show first way orange
    histogram=cv2.calcHist([self.__src],[0],None,[256],[0,256])#gray
    plt2.figure(figsize=(8,8))
    plt2.xlim((0,255))
    plt2.plot(histogram)

    #show secound way orange
    plt2.hist(self.__src.ravel(),256,[0,256])
    plt2.show()



image1=Image_process('/content/FB_IMG_1696878202893.jpg')
image1.image_filter_mode(5)
# image1.image_color_mode()
image1.draw_histogram()

image2=Image_process('/content/473163988_580254778184636_8779327347367497735_n.jpg')
image2.image_filter_mode(5)
# image1.image_color_mode()
image2.draw_histogram()

image3=Image_process('/content/IMG_20241216_175701_898.jpg')
image3.image_filter_mode(5)
# image1.image_color_mode()
image3.draw_histogram()




#negarive mode
#255-matirex

#blur
#GaussianBlur
# or in imread add cv2.blur

#gray
# in imread IMREAD.GRAY

#shrink
# cv2.convertScaleAbs(,alphe=.5,beta=0)

#slid
# cv2.convertScaleAbs(,alphe=1,beta=50)darke
# cv2.convertScaleAbs(,alphe=1,beta=-50)white

#streatch
# cv2.convertScaleAbs(,alphe=1.5,beta=0)




#in opencv not found any method for draw histogram but can calculate histogram by method that method is  calcHist

# 1
#image as matrix in array [imgmatrix]

# 2
#dims (القناة/القنوات)
#dims or channal b g r 0 1 2 for rgb or 0 for gray in array [1]

# 3
# mask (تقييد الحساب لمنطقة معينة)
#mask this None for calculate histgrame for fullimage but you want calculate for spcific place in image can do that
# take spcific place for calculate histogram in it

# 4
#bins capacity of bixl [256]
# this is capacity for tabe

# 5
# range (مدى القيم)
# the range will search on it how image is contean 0 to 256 if your range from 0 to 128 this mean you must show all pixal have this values
#range of bixal must be filter and calculate in histogram [0-256]

#range = bins1 union bins2 unoin bins3 and so on for 255
#bin1=[0,15]; bin2=[16,31] we incease 16 value in bixal for 255 bins is group of bixals





################# for draw histogram ###############

#1 Short Way : use Matplotlib plotting functions
#2 Long Way : use OpenCV drawing functions

#this way is very good if have gray color
# 1 pyplot from matplotlib and function plt.hist(matreix image .reval(),bins,range)
# 2 from open cv call calcHist([image matrex],[dims],mask,[bins],[range]) then add this in plt.plot from matplotlib

############ note ###########
# now we now thing in cv2 imread take url of image
# then any module in cv2 deal with matrix of image as imshow() and any module deal with matrix as first parameter
# kernal what is it => kernal is matrix add on origin matrix or matrix of image
# why because this matrix deal with each bixal in origin image change his futrue by spcific calculation




"""
link function plot
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

link function hist
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

matplotlib.pyplot.hist(x,
 bins=None,
 *,
 range=None,
 density=False,
 weights=None,
 cumulative=False,
 bottom=None,
 histtype='bar',
 align='mid',
 orientation='vertical',
 rwidth=None,
 log=False,
 color=None,
 label=None,
 stacked=False,
 data=None,
 **kwargs)

"""


"""
cv2.convertScaleAbs() function, which allows you to adjust the brightness and contrast using a combination of scaling and shifting the pixel values

# Adjust the brightness and contrast
# g(i,j)=α⋅f(i,j)+β
# control Contrast by 1.5
alpha = 1.5
# control brightness by 50
beta = 50
image2 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
"""


# some kernals
"""
blur()
boxFilter()
GaussianBlur()
medianBlur()
bilateralFilter()
Sobal()
Laplacian()
Canny()
getDerivKernels()
Scharr() better than Sobal()
Laplacian()
getStructuringElement() cv2.MORPH_RECT → مستطيل, cv2.MORPH_ELLIPSE → بيضاوي, دائريcv2.MORPH_CROSS → شكل الصليب
filter2D(src, ddepth, kernel) if you want custom kernal for yourself


التنعيم (Blurring/Smoothing): Box, Gaussian, Median, Bilateral
الكشف عن الحواف (Edge Detection): Sobel, Scharr, Laplacian
المورفولوجي (Morphological kernels): Rect, Ellipse, Cross
المخصص (Custom): أي كيرنل أنت تعرّفه
"""

