# image-processing
## 1.Develop a program to display grayscale image using read and write operation.
Grayscale
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

### output

