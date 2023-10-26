# NeRF-LiDAR-cGAN

This is the working repo for the following paper:

Ming-Fang Chang, Akash Sharma, Michael Kaess, and Simon Lucey. Neural Radiance Fields with LiDAR Maps. ICCV 2023.
[paper link](https://openaccess.thecvf.com//content/ICCV2023/papers/Chang_Neural_Radiance_Field_with_LiDAR_maps_ICCV_2023_paper.pdf)

<p align="center">
<img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/block1_v2.png' width='640' align=”center”>
</p>

If you find our work useful, please consider to cite:

```
@inproceedings{Chang2023iccv,
	title={{Neural Radiance Fields with LiDAR Maps}},
	author={Ming-Fang Chang and Akash Sharma and Michael Kaess and and Simon Lucey},
	booktitle=International Conference on Computer Vision,
	year={2023} 
} 
```



### 1. Environment
   1. The code was implemented and tested with python 3.7, `PyTorch v1.12.1` and `DGL 0.9.v1post1`.

### 2. Download the datasets:
   1. Training/val samples (preprocessed DGL graphs). [preprocessed graphs](https://drive.google.com/drive/folders/1svkEFQQmVKdrpSgjQrvY8zamw7q8xIlq?usp=drive_link) The preprocessed DGL graphs contain geometric information needed for volume rendering (see 3. for visualizations).
   2. The LiDAR point cloud maps. [maps](https://drive.google.com/file/d/1q9K-n6QHhz7Y55e2mGBrEvaWNiveuMpf/view?usp=drive_link)
   3. Other dataset information (ground truth images, camera poses, etc). [dataset](https://drive.google.com/file/d/1o8kKnhwCoDAckpOn0mxlVt1xRmc69VG4/view?usp=drive_link)
   4. Masks for dynamic objects. [masks](https://drive.google.com/file/d/1DjfOWRjlplTpANvS4YBebp78WtHpDgRJ/view?usp=drive_link)
   5. Specify your local data folder path in `configs/config.ini`, or make a symlink named `data` pointing to your dataset folder.
  
### 3. Visualize the data:
   1. The visualization code was tested with `pyvista v0.37.0`.
   3. Run `python3 visualize_data.py --log_id=<log_id> --name_data=clean`
   4. Expected outputs include (from log `2b044433-ddc1-3580-b560-d46474934089`):
      1. Camera rays (black), ray samples (red), and nearby LiDAR points (green) of subsampled pixels.
      2. GT rgb and depth.
      3. Train (blue) / val (red) camera poses on the map.
<p align="center">
   <img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/ev_1.gif' height='320'>
</p>
<p align="center">
   <img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/ev_2.png' height='200'>
<img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/ev_3.png' height='200'>
</p>

### 4. Run the code:
   1. Run `python3 train.py --name_data=clean --log_id=<log_id> --name_config=config.ini --eval_only`.
   2. Check the results with tensorboard (e.g. Run `tensorboard --logdir=logs>` to see the visuals. The log path can be specified in `configs/config.ini`).
   3. You can download the trained weights from [weights (clean maps)](https://drive.google.com/file/d/1ZjEXD1XigYyJwazdJ8PTD6uM6ell5Xyb/view?usp=drive_link) [weights (noisy maps)](https://drive.google.com/file/d/1J_G54UECEhBivVDQMVBMo2uslv6GUr7b/view?usp=sharing).
   4. Expected outputs (from log `2b044433-ddc1-3580-b560-d46474934089`):
<p align="center">
   <img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/demo_1.png' height='200'>
   <img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/demo_2.png' height='200'>
   <img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/demo_3.png' height='200'>
   <img src='https://github.com/alliecc/NeRF-LiDAR-cGAN/blob/main/imgs/demo_4.png' height='200'>
</p>

   5. For netowrk training, remove the `--eval_only` argument. 
   

### 

