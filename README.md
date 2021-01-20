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

