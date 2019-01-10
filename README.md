# SegNet-Implementation
The Davis 2016 dataset is used where outputs are in binary segmentation. 

The pixel-wise loss function was implemented in this. 


http://latex.codecogs.com/svg.latex?%5C%5BL%28X_%7Bt%7D%29+%3D+-%281-w%29+%5Csum_%7Bi%2Cj%5Cepsilon+fg%7D+log+E%28y_%7Bi%2Cj%7D%3D1%2C%5CTheta+%29-%28w%29+%5Csum_%7Bi%2Cj%5Cepsilon+bg%7D+log+E%28y_%7Bi%2Cj%7D%3D0%2C%5CTheta+%29%5C%5D

The IoU metric is also implemented.
