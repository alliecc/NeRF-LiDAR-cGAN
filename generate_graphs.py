#this code is for generating the preprocessed DGL graphs
#
import dgl
import configparser
import argparse
import os
import torch
import pickle
import open3d as o3d
import numpy as np
from src.trainer import Trainer
from src.dataloader import LiDARDataset, collate_fn
from src.utils import get_ray_dir
from pykdtree.kdtree import KDTree

class GraphGenerator(LiDARDataset):
    def __init__(self, cfg, split, name_data):
        super().__init__(cfg, split, name_data)

        self.ray_sample_steps = int(cfg['ray_sample_steps'])


        path_lidar_map = os.path.join(cfg['path_data_folder'], 'argoverse_2_maps', f"{cfg['log_id']}_{cfg['name_data']}.ply")

        print('Load lidar map from', path_lidar_map)
        #The point cloud map was quantized beforehand. We don't need overly dense anchor LiDAR points.
        self.map = np.asarray(o3d.io.read_point_cloud(path_lidar_map).points)
        self.knn_radius = 0.15 #m
        self.ground_radius = 3
        self.max_num_map_points = 10 #the map points to aggregate features from
        self.num_sample_per_ray = 16
        self.depth_min = 1
        self.depth_max = 70
        self.map_kd_tree = KDTree(self.map)


        #for argoverse datasets, we can use the ground labels from the map for better ground sampling
        path_ground_mask = path_lidar_map[:-4] + '_ground_labels.pkl'
        print('Read argoverse ground height maps from ', path_ground_mask)
        if not os.path.exists(path_ground_mask):
            from av2.map.map_api import ArgoverseStaticMap, GroundHeightLayer
            from pathlib import Path
            import glob

            log_map_dirpath = os.path.join(cfg['path_data_folder'], 'argoverse_2_hd_maps', cfg['log_id'])
            log_map_path = Path(glob.glob(log_map_dirpath+'/log_map_*.json')[0])

            avm = ArgoverseStaticMap.from_json(log_map_path)
            avm.raster_ground_height_layer = GroundHeightLayer.from_file(Path(log_map_dirpath))
            self.map_ground_mask = avm.get_ground_points_boolean(self.map)
    
            with open(path_ground_mask, 'wb') as f:
                pickle.dump(self.map_ground_mask, f)
        else:
            with open(path_ground_mask, 'rb') as f:
                self.map_ground_mask = pickle.load(f)


    def __getitem__(self, ind):

        path_graph = os.path.join(self.cfg['path_preprocessed_graph'], self.name_data, self.cfg['log_id'], f'{self.split}_{ind:04d}_v2.bin')

        if not os.path.exists(path_graph):
            print(f'Generate graph and save to {path_graph}')
            E = torch.from_numpy(self.data['E'][ind]).float()
            K = torch.from_numpy(self.data['K'][ind]).float()
            graph = self.generate_dgl_graph_from_E(E, K)
            dgl.data.utils.save_graphs(path_graph, graph)

        else:
            graph = dgl.load_graphs(path_graph)[0][0]


    def generate_dgl_graph_from_E(self, E, K):
        u, v = torch.meshgrid(torch.arange(self.img_size), torch.arange(self.img_size))
        uv = torch.stack((u, v)).reshape(2, -1).transpose(0, 1)
        ray_dirs = get_ray_dir(uv, K, E)

        cam_center = torch.inverse(E)[0:3, 3]

        sample = self.point_sampler(cam_center, ray_dirs, E, K)

        return self.build_dgl_graph(sample)     


    def generate_initial_ray_sample(self, cam_center, ray_dirs):

        #the integer steps need to be converted to actual depth values when computing depth
        steps = torch.arange(self.ray_sample_steps, dtype=torch.short)[None, :]

        steps_dist_to_cam = self.depth_min + self.depth_max * steps/self.ray_sample_steps
        sample_pts = cam_center[None, None, :] + steps_dist_to_cam[:, :, None]*ray_dirs[:, None, :]

        return sample_pts, steps 

    def point_sampler(self, cam_center, ray_dirs, E, K):
        num_pixels = ray_dirs.shape[0]
        
        sample0_xyz, steps0 = self.generate_initial_ray_sample(cam_center, ray_dirs)  # N0xM0x3,  N0xM0x1

        N0, M0, _ = sample0_xyz.shape
        sample0_xyz = sample0_xyz.reshape(-1, 3) # (N0*M0) x3 in world coorinate frame


        #remove the samples outside the map range
        map_min = torch.from_numpy(self.map.min(axis=0))
        map_max = torch.from_numpy(self.map.max(axis=0))
        mask_sample_valid = ((sample0_xyz > map_min) * (sample0_xyz < map_max)).all(axis=1)

        #only keep the samples whose distance to closest map points is less than knn_radius
        dist_sample_to_map_points, nearest_map_points_idx = self.map_kd_tree.query(sample0_xyz[mask_sample_valid].numpy(), distance_upper_bound=self.knn_radius)

        mask_knn = dist_sample_to_map_points < self.knn_radius 
        
        dist_sample_to_map_points, nearest_map_points_idx = self.map_kd_tree.query(sample0_xyz[mask_sample_valid][mask_knn].numpy(), k = self.max_num_map_points)
        nearest_map_points_idx = torch.from_numpy(nearest_map_points_idx.astype(int)).long()


        #check if it's sorted by distance....

        #keep only up to num_sample_per_ray for each pixel


        #keep only up to max_num_map_points for each ray sample
        self.num_sample_per_ray

        #put everything into a dgl graph format


    def build_dgl_graph(self, sample):
        pass



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_id',  type=str, required=True)
    parser.add_argument('--name_config',  type=str,  required=True)
    parser.add_argument('--name_data', choices=['clean', 'noisy'], type=str, required=True)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    print(args)

    cfg = configparser.ConfigParser()
    cfg.read(os.path.join('configs', args.name_config))
    cfg = cfg['DEFAULT']

    print(cfg)
    cfg['log_id'] = str(args.log_id)
    cfg['name_data'] = str(args.name_data)
    cfg['device'] = 'cpu' #Use CPU to generate graphs by default



    list_splits = ['train', 'val']
    for split in list_splits:
        dataset = GraphGenerator(cfg, split, cfg['name_data'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        for ind, batch in enumerate(dataloader):
            print(f'Processing sample {split}:{ind:04d}')


