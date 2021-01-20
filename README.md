# image-processing

## 1.Develop a program to display grayscale image using read and write operation.
## Grayscale
Grayscale is a range of monochromatic shades from black to white. Therefore, a grayscale image contains only shades of gray and no color.Grayscale is a range of shades of gray without apparent color. The darkest possible shade is black, which is the total absence of transmitted or reflected light.

### code
import cv2
import numpy as np
image = cv2.imread(&#39;cat.jpg&#39;)
image = cv2.resize(image, (0, 0), None, 1.00, 1.00)
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
cv2.imshow(&#39;cat&#39;, numpy_horizontal_concat)
cv2.waitKey()

**output**
![image](https://user-images.githubusercontent.com/72584581/105153465-523d4d00-5abd-11eb-911d-3f3b8b7b8da8.png)

## 2. Develop a program to perform linear transformation on image. (Scaling and rotation)
## Scaling
In computer graphics and digital imaging, image scaling refers to the resizing of a digital image.
### code
import cv2 as c
img=c.imread("img3.jpg")
c.imshow('image',img)
nimg=c.resize(img,(0,0),fx=0.50,fy=0.50)
c.imshow("Result",nimg)
c.waitKey(0)

**output**
![image](https://user-images.githubusercontent.com/72584581/105157710-19ec3d80-5ac2-11eb-8832-9774043a988d.png)
![image](https://user-images.githubusercontent.com/72584581/105157890-44d69180-5ac2-11eb-8b6b-e3f005908190.png)

## Rotation
Images can be rotated to any degree clockwise or otherwise. We just need to define rotation matrix listing rotation point, degree of rotation and the scaling factor.
### code
import cv2 
import numpy as np 
img = cv2.imread('nature.jpg') 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 120, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('image', img)
cv2.waitKey(0) 
cv2.imshow('result',res) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

**output**
![image](https://user-images.githubusercontent.com/72584581/105158573-f7a6ef80-5ac2-11eb-8e0c-15b98e81be46.png)
![image](https://user-images.githubusercontent.com/72584581/105158699-1e652600-5ac3-11eb-9528-4c4e305f221c.png)

## 3. Develop a program to find sum and mean of a set of images.
Create n number of images and read the directory and perform operation.

os.listdir() - returns a list containing the names of the entries in the directory given by path. sum - add a constant value to an image. mean - will give you an idea of what pixel color to choose to summarize the color of the complete image.
### code
import cv2
import os
path = "F:\image"
imgs=[]
dirs=os.listdir(path)
for file in dirs:
    fpat=path+"\\"+file
    imgs.append(cv2.imread(fpat))
i=0
for im in imgs:
    i=i+1
print(i)
cv2.imshow('sum',len(im))
cv2.imshow('mean',len(im)/im)
cv2.waitKey(0)

**output**
![image](https://user-images.githubusercontent.com/72584581/105160530-32aa2280-5ac5-11eb-8973-e1a787c845aa.png)
![image](https://user-images.githubusercontent.com/72584581/105160714-6be29280-5ac5-11eb-8081-2d7ff1836e3d.png)

## 4.Convert color image to Gray scale and binary image
##Gray scale image 
  Gray scale is simply one in which the only colors are shades of gray. binary image - is one that consists of pixels that can have one of exactly two colors, usually black and white. cv2.threshold() - is the assignment of pixel values in relation to the threshold value provided.
### code
import cv2
image=cv2.imread("cat.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(tresh,blackAndWhiteImage)=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("gray",gray)
cv2.imshow("BINARY",blackAndWhiteImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
**output**
![image](https://user-images.githubusercontent.com/72584581/105161421-37bba180-5ac6-11eb-9680-131a7b4d5211.png)
![image](https://user-images.githubusercontent.com/72584581/105161544-57eb6080-5ac6-11eb-878d-4a5ff64e45c3.png)

## 5.Develop a program to convert color image into different color space.
Color Space - is a specific organization of colors.
###code
import cv2
image=cv2.imread("nature.jpg")
cv2.imshow("old",image)
cv2.waitKey()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",hsv)
cv2.waitKey(0)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB",lab)
cv2.waitKey(0)
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imshow("HLS",hls)
cv2.waitKey(0)
yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
cv2.imshow("YUV",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

**output**
![image](https://user-images.githubusercontent.com/72584581/105162204-258e3300-5ac7-11eb-9b95-9236b1d87059.png)
![image](https://user-images.githubusercontent.com/72584581/105162301-49517900-5ac7-11eb-8747-b8aab7e41a6f.png) 
