# SegNet-Implementation
The Davis 2016 dataset is used where outputs are in binary segmentation. 

The pixel-wise loss function was implemented in this. 

\[L(X_{t}) = -(1-w) \sum_{i,j\epsilon fg} log E(y_{i,j}=1,\Theta )-(w) \sum_{i,j\epsilon bg} log E(y_{i,j}=0,\Theta )\]

The IoU metric is also implemented.
