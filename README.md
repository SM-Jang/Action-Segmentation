This is a implementation of TCF using previous state-of-the-art backbone models in Action-Segmentation.
![curvature](https://user-images.githubusercontent.com/74584105/160565726-80c4d61b-b79a-42fa-ba46-a1e03b18b5a9.png)
1. Installation
- python 3.7.10  
- pytorch=='1.5.0'

2. Dataset:
GTEA, 50salads, Breakfast
we used I3D features from https://github.com/yabufarha/ms-tcn

3. Backbone:

MS-TCN, MS-TCN++, ASRF, ASFormer
- MS-TCN   https://github.com/yabufarha/ms-tcn
- MS-TCN++ https://github.com/sj-li/MS-TCN2
- ASRF     https://github.com/yiskw713/asrf
- ASFormer https://github.com/ChinaYi/ASFormer

4. Result
|GTEA|내용|설명|
|:---:|:---:|:---:|
|Method|F1@10|F1@25|F1@50|Edit|Acc|
|MS-TCN|87.5|85.4|74.6|81.4|79.2|
|MS-TCN(our impl.)|87.5|85.4|74.6|81.4|79.2|
|MS-TCN + TCF|87.5|85.4|74.6|81.4|79.2|
|Gain|87.5|85.4|74.6|81.4|79.2|
