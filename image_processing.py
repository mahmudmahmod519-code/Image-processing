from enum import Enum
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import pandas as pd
from glob import glob



class Image_process:

  class Mode(Enum):
    NEGATIVE_MODE=-1
    SHRINK_MODE=0
    # SLID_MODE=1
    BULR_MODE=1
    STRITCH_MODE=2
    DAREKER_MODE=3
    WHITER_MODE=4
    GRAY_MODE=5

  dev_env=False

  def __init__(self,src,for_dev=False):
    if(for_dev):
      dev_env=True
      self.__src=cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    else:
      self.__src=cv2.imread(src)

  def show_image(self):
    cv2_imshow(self.__src)


  def image_filter_mode(self,mode):
    if(mode==self.Mode.NEGATIVE_MODE.value):
      self.__src=255-self.__src
    elif(mode==self.Mode.SHRINK_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=.5,beta=0)

    elif(mode==self.Mode.STRITCH_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=1.5,beta=0)

    elif(mode==self.Mode.WHITER_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=1,beta=50)

    elif(mode==self.Mode.DAREKER_MODE.value):
      self.__src=cv2.convertScaleAbs(self.__src,alpha=1,beta=-50)

    elif(mode==self.Mode.BULR_MODE.value):
      self.__src=cv2.blur(self.__src,(10,20))

    elif(mode==self.Mode.GRAY_MODE.value):
      self.__src=cv2.cvtColor(self.__src,cv2.COLOR_RGB2GRAY)


  def draw_histogram(self):
    plt2.figure(figsize=(8,8))
    plt2.xlim((0,255))

    if(self.dev_env):
      # show first way orange
      histogram=cv2.calcHist([self.__src],[0],None,[256],[0,256])#gray
      plt2.plot(histogram)
    else:
      for i,col in enumerate(['r','g','b']):
        histogram=cv2.calcHist([self.__src],[i],None,[256],[0,256])
        plt2.plot(histogram,color=col)
    #show secound way orange
    # plt2.hist(self.__src.ravel(),256,[0,256])
    plt2.show()



image1=Image_process('/content/1000129864.jpg')
image1.image_color_mode()
# image1.image_color_mode()
image1.draw_histogram()

image2=Image_process('/content/473224271_580251011518346_3843995772034379473_n.jpg')
image2.image_filter_mode()
# image1.image_color_mode()
image2.draw_histogram()

image3=Image_process('/content/٢٠٢٠١٢٢٠_٠٠٣١٤٣.jpg')
image3.image_filter_mode()
# image1.image_color_mode()
image3.draw_histogram()

