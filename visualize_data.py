import torch
import configparser
import matplotlib
import cv2
import os
import argparse
import open3d as o3d
import numpy as np
import pyvista as pv
from src.dataloader import get_data_loader

color_map = matplotlib.colormaps['viridis']


def visualize_a_sample(sample, map_lidar):

    dyn_mask = sample['img_dyn_mask']
    img_gt = sample['img_rgb_gt']

    img_size, _, _ = img_gt.shape

    g = sample['graph']
    depth_lidar = g.nodes['pixel'].data['depth_lidar'][:, 0]
    ind_pixel = g.nodes['pixel'].data['ind_pixel']

    xyz_ray_sample = g.nodes['sample'].data['sample_xyz'].numpy()
    xyz_lidar_pt = map_lidar[g.nodes['voxel'].data['ind_voxel']]
    cam_center = sample['cam_center'].numpy()

    edges_sample_pixel = g[('sample', 'ray', 'pixel')].edges()
    edges_voxel_sample = g[('voxel', 'neighbor', 'sample')].edges()

    print('Visualize a graph in 3D...this might take a while (red: ray samples, green: aggrgated LiDAR points).')
    pl = pv.Plotter()
    pl.set_background('white')
    pl.add_mesh(pv.PolyData(xyz_ray_sample), 'gray', render_points_as_spheres=True, point_size=5,  opacity=0.02)
    pl.add_mesh(pv.PolyData(xyz_lidar_pt), 'k', render_points_as_spheres=True, point_size=2,  opacity=1)
    pl.add_mesh(pv.PolyData(cam_center), 'b', render_points_as_spheres=True, point_size=50,  opacity=0.8)

    downsampling_ratio = 1000

    ind_pixel_vis = np.arange(0, len(ind_pixel), downsampling_ratio)
    list_invalid_pixel = []
    for ind in ind_pixel_vis:
        ind_ray_sample = edges_sample_pixel[0][edges_sample_pixel[1] == ind]

        for j in ind_ray_sample:
            pl.add_mesh(pv.Line(xyz_ray_sample[j], cam_center), color=(1, 0, 0), line_width=1, opacity=0.5)

        for j in ind_ray_sample:
            ind_lidar_pt_vis = edges_voxel_sample[0][edges_voxel_sample[1] == j]
            for jj in ind_lidar_pt_vis:
                pl.add_mesh(pv.Line(xyz_lidar_pt[jj], xyz_ray_sample[j]), color=(1, 0, 0), line_width=1, opacity=0.5)

            pl.add_mesh(pv.PolyData(xyz_lidar_pt[ind_lidar_pt_vis]), 'g',
                        render_points_as_spheres=True, point_size=5,  opacity=1)

        pl.add_mesh(pv.PolyData(xyz_ray_sample[ind_ray_sample]), 'r',
                    render_points_as_spheres=True, point_size=5,  opacity=1)

    pl.show()

    print('Visualize the ground truth RGB and LiDAR depth.')
    depth_min = 0.3
    img_depth_lidar_inv = torch.zeros(img_size**2)
    d_inv = 1/depth_lidar[depth_lidar > depth_min]
    d_sel = img_depth_lidar_inv[ind_pixel]
    d_sel[depth_lidar > depth_min] = 1/depth_lidar[depth_lidar > depth_min]

    img_depth_lidar_inv[ind_pixel] = d_sel
    img_depth_lidar_inv = color_map(img_depth_lidar_inv*3)[:,  0:3]
    img_depth_lidar_inv[img_depth_lidar_inv == 0] = 0

    img_to_show = np.zeros((img_size, img_size*2, 3))
    img_to_show[:, 0:img_size] = (img_gt.numpy()+1)/2
    img_to_show[:, img_size:] = img_depth_lidar_inv.reshape(img_size, img_size, 3)
    cv2.imshow('RGB and depth GT', img_to_show[:, :, [2, 1, 0]])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_cam_pose_on_map(dataloader_val, dataloader_train, map_lidar):

    print('Visualize the ground truth camera poses (blue: training, red: testing).')
    pose_val = np.linalg.inv(dataloader_val.dataset.data['E'])
    pose_train = np.linalg.inv(dataloader_train.dataset.data['E'])
    pl = pv.Plotter()
    pl.set_background('white')
    pl.add_mesh(pv.PolyData(map_lidar), 'gray', render_points_as_spheres=True, point_size=1,  opacity=0.5)
    pl.add_mesh(pv.PolyData(pose_val[:, 0:3, 3]), 'r', render_points_as_spheres=True, point_size=10,  opacity=0.5)
    pl.add_mesh(pv.PolyData(pose_train[:, 0:3, 3]), 'b', render_points_as_spheres=True, point_size=5,  opacity=1)
    pl.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id',  type=str, required=True)
    parser.add_argument('--name_data', choices=['clean', 'noisy'], type=str, required=True)

    args = parser.parse_args()
    print(args)

    cfg = configparser.ConfigParser()
    cfg.read('configs/config.ini')
    cfg = cfg['DEFAULT']
    print(cfg)
    cfg['log_id'] = str(args.log_id)

    dataloader_train = get_data_loader(cfg, args.name_data, 'train')
    dataloader_val = get_data_loader(cfg, args.name_data, 'val')

    path_lidar_map = os.path.join(cfg['path_data_folder'], 'argoverse_2_maps', f"{cfg['log_id']}_{args.name_data}.ply")
    map_lidar = np.asarray(o3d.io.read_point_cloud(path_lidar_map).points)

    # to visualize the graph, LiDAR depth, ground truth image for one single sample
    sample_ind_to_visualize = 7
    visualize_a_sample(dataloader_val.dataset[sample_ind_to_visualize], map_lidar)

    # to visualize the camera poses on the map of the whole dataset
    visualize_cam_pose_on_map(dataloader_val, dataloader_train, map_lidar)
