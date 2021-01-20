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
![image](https://user-images.githubusercontent.com/72584581/105162434-7867ea80-5ac7-11eb-977b-9cd54f549bce.png)
![image](https://user-images.githubusercontent.com/72584581/105162554-a0efe480-5ac7-11eb-8b19-e5d0acbff4b2.png)
![image](https://user-images.githubusercontent.com/72584581/105162743-dd234500-5ac7-11eb-9cbc-dd7c01d2c914.png)

## 6.Develop a program to create an image from 2D array.
np.linspace() - is an in-built function in Python's NumPy library. It is used to create an evenly spaced sequence in a specified interval. Image.fromarray() - This function converts a numerical (integer or float) numpy array of any size and dimensionality into a CASA image. np.reshape() - function shapes an array without changing data of array.
### code
import numpy as np
from PIL import Image
import cv2 as c 
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [150, 128, 0]
array[:,100:] = [0, 0, 255]   
img = Image.fromarray(array)
img.save('nature.jpg')
img.show()
c.waitKey(0)

**output**
![image](https://user-images.githubusercontent.com/72584581/105163313-9c77fb80-5ac8-11eb-9423-a924c187c73b.png)

## 7.program to find the neighbor of matrix.

X = [[1,2,3], [4 ,5,6], [7 ,8,9]] Y = [[9,8,7], [6,5,4], [3,2,1]] result = [[0,0,0], [0,0,0], [0,0,0]] for i in range(len(X)):
for j in range(len(Y)): result[i][j] = X[i][j] + Y[i][j] print("Resultant array:") for r in result: print(r) def neighbors(radius, rowNumber, columnNumber): return [[result[i][j] if i >= 0 and i < len(result) and j >= 0 and j < len(result[0]) else 0 for j in range(columnNumber-1-radius, columnNumber+radius)] for i in range(rowNumber-1-radius, rowNumber+radius)] neighbors(4,2,2)

OUTPUT: ![image](https://user-images.githubusercontent.com/72584581/105164697-4a37da00-5aca-11eb-913d-35e1f77f6315.png)

## 8.Program to find the Sum of neighbour value of Matrix.
import numpy as np M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] M = np.asarray(M) N = np.zeros(M.shape) def sumNeighbors(M,x,y): l = [] for i in range(max(0,x-1),x+2):
for j in range(max(0,y-1),y+2): try: t = M[i][j] l.append(t) except IndexError: pass return sum(l)-M[x][y] for i in range(M.shape[0]): for j in range(M.shape[1]): N[i][j] = sumNeighbors(M, i, j) print ("Original matrix:\n", M) print ("Summed neighbors matrix:\n", N)

Output: ![image](https://user-images.githubusercontent.com/72584581/105164960-997e0a80-5aca-11eb-9343-cb5f2d38e1e0.png)

## 9.Operator Overloading in C++:Assignment operator of 2 Matrix.

C++ has the ability to provide the operators with a special meaning for a data type, this ability is known as operator overloading. Assignment operator - are used to assigning value to a variable. #include int findSum(int n) { // Generate matrix int a[100][100],b[100][100]; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) std::cin>>a[i][j] ; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) b[i][j]=a[i][j]; // Compute sum int sum = 0; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) sum += b[i][j]; std::cout <<"sum of elements: ";

return sum; } int main() { int n = 3; std::cout << findSum(n) ; return 0;

}

Ouput: image

Describe: (i) Anaconda : Anaconda is a distribution of the Python and R programming languages for scientific computing , that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS. Anaconda is popular because it brings many of the tools used in data science and machine learning with just one install, so it's great for having short and simple setup. Like Virtualenv, Anaconda also uses the concept of creating environments so as to isolate different libraries and versions. (ii)Spyder : Spyder is an open source cross-platform integrated development environment (IDE) for scientific programming in the Python language. Spyder integrates with a number of prominent packages in the scientific Python stack, including NumPy, SciPy, Matplotlib, pandas, IPython, SymPy and Cython, as well as other open source software. Spyder, the Scientific Python Development Environment, is a free integrated development environment (IDE) that is included with Anaconda. It includes editing, interactive testing, debugging, and introspection features. ... Spyder is also pre-installed in Anaconda Navigator, which is included in Anaconda. (iii)Jupiter : The Jupyter Notebook application allows you to create and edit documents that display the input and output of a Python or R language script. Once saved, you can share these files with others. NOTE: Python and R language are included by default, but with customization, Notebook can run several other kernel environments.

