This is a implementation of TCF using previous state-of-the-art backbone models in Action-Segmentation.
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
![gtea](https://user-images.githubusercontent.com/74584105/160565755-ecb55bc0-1b96-490a-930d-251915699156.png)
![50salads](https://user-images.githubusercontent.com/74584105/160565758-43fd733d-cceb-4a0b-a1e0-454f458050d5.png)
![breakfast](https://user-images.githubusercontent.com/74584105/160565760-fba120fe-6d4b-4637-9922-f27ed36c9d0b.png)

5. Effect
![curvature](https://user-images.githubusercontent.com/74584105/160565726-80c4d61b-b79a-42fa-ba46-a1e03b18b5a9.png)
