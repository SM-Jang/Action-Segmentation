# TCF: a new feature synthes framework in action segmentation
- A novel video feature enhancement
- No additional temporal information
- Relieve under-segmentation and correct misprediction

![effect](https://user-images.githubusercontent.com/74584105/160569533-1fd845f0-ff69-46fb-a73d-c47e961d82b7.png)

# 1. Installation
- python 3.7.10  
- pytorch=='1.5.0'

# 2. Dataset(GTEA, 50salads, Breakfast)
![dataset](https://user-images.githubusercontent.com/74584105/160568902-23d72990-319b-4d9d-bde0-db35842f7f0b.png)
we used I3D features from https://github.com/yabufarha/ms-tcn

# 3. Backbone:
Most of the codes are copied from 
- [MS-TCN](https://github.com/yabufarha/ms-tcn)
- [MS-TCN++](https://github.com/sj-li/MS-TCN2)
- [ASRF](https://github.com/yiskw713/asrf)
- [ASFormer](https://github.com/ChinaYi/ASFormer)

# 4. Framework
![model](https://user-images.githubusercontent.com/74584105/160568800-aeb4cfd4-9d9c-43b4-83c1-65cedc25d264.jpg)


# 5. Result Table
![그림9](https://user-images.githubusercontent.com/74584105/160568464-703a878c-1864-4d23-8083-161a2337b523.png)

# 6. Effect analysis
![curvature](https://user-images.githubusercontent.com/74584105/160565726-80c4d61b-b79a-42fa-ba46-a1e03b18b5a9.png)
