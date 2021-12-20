This is a implementation of TCF using previous state-of-the-art backbone models in Action-Segmentation.

python 3.7.10  pytorch=='1.5.0'

Dataset:
GTEA, 50salads, Breakfast
we used I3D features from https://github.com/yabufarha/ms-tcn

Backbone:
MS-TCN, MS-TCN++, ASRF, ASFormer

Most of the codes are copied from 

MS-TCN   https://github.com/yabufarha/ms-tcn
MS-TCN++ https://github.com/sj-li/MS-TCN2
ASRF     https://github.com/yiskw713/asrf
ASFormer https://github.com/ChinaYi/ASFormer