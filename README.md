This is a implementation of TCF using previous state-of-the-art backbone models in Action-Segmentation.
0. Insight
![effect](https://user-images.githubusercontent.com/74584105/160568664-cf9287b2-3092-45eb-aa09-4e778f5ec538.jpg)

1. Installation
- python 3.7.10  
- pytorch=='1.5.0'

2. Dataset(GTEA, 50salads, Breakfast)
![dataset](https://user-images.githubusercontent.com/74584105/160568902-23d72990-319b-4d9d-bde0-db35842f7f0b.png)
we used I3D features from https://github.com/yabufarha/ms-tcn

3. Backbone:
MS-TCN, MS-TCN++, ASRF, ASFormer
- [MS-TCN](https://github.com/yabufarha/ms-tcn)
- [MS-TCN++](https://github.com/sj-li/MS-TCN2)
- [ASRF](https://github.com/yiskw713/asrf)
- [ASFormer](https://github.com/ChinaYi/ASFormer)

4. Framework
![model](https://user-images.githubusercontent.com/74584105/160568800-aeb4cfd4-9d9c-43b4-83c1-65cedc25d264.jpg)


5. Result Table
![그림9](https://user-images.githubusercontent.com/74584105/160568464-703a878c-1864-4d23-8083-161a2337b523.png)

6. Effect analysis
![curvature](https://user-images.githubusercontent.com/74584105/160565726-80c4d61b-b79a-42fa-ba46-a1e03b18b5a9.png)
