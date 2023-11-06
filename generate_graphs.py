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

        return self.point_sampler_build_dgl_graph(sample)     


    def generate_initial_ray_sample(self, cam_center, ray_dirs):

        #the integer steps need to be converted to actual depth values when computing depth
        steps = torch.arange(self.ray_sample_steps, dtype=torch.short)[None, :]

        steps_dist_to_cam = self.depth_min + self.depth_max * steps/self.ray_sample_steps


        sample_pts = cam_center[None, None, :] + steps_dist_to_cam[:, :, None]*ray_dirs[:, None, :]

        return sample_pts, steps 

    def point_sampler_build_dgl_graph(self, cam_center, ray_dirs, E, K):
        num_pixels = ray_dirs.shape[0]
        
        sample0_xyz, steps0 = self.generate_initial_ray_sample(cam_center, ray_dirs)  # N0xM0x3,  N0xM0x1

        N0, M0, _ = sample0_xyz.shape
        sample0_xyz = sample0_xyz.reshape(-1, 3).numpy() # (N0*M0) x3 in world coorinate frame


        #remove the samples outside the map range
        map_min = self.map.min(axis=0)
        map_max = self.map.max(axis=0)
        ind_sel_samples = np.arange(N0*M0)
        ind_sel_samples = ind_sel_samples[((sample0_xyz > map_min) * (sample0_xyz < map_max)).all(axis=1)]

        #only keep the samples whose distance to closest map points is less than knn_radius
        dist_sample_to_map_points, nearest_map_points_ind = self.map_kd_tree.query(sample0_xyz[ind_sel_samples], distance_upper_bound=self.knn_radius)
        mask_knn = dist_sample_to_map_points < self.knn_radius 

        #select only num_sample_per_ray samples for each pixel
        ind_sel_samples = ind_sel_samples[mask_knn]
        mask_all_samples = np.zeros(N0*M0)
        mask_all_samples[ind_sel_samples] = 1
        mask_all_samples = mask_all_samples.reshape(N0, M0)
        cum_sum_num_samples_per_pixel = mask_all_samples.cumsum(axis=1)
        mask_all_samples[cum_sum_num_samples_per_pixel > self.num_sample_per_ray] = 0
        mask_all_samples = (mask_all_samples==1).flatten()

        #query knn neighbors for the valid samples
        sample = sample0_xyz[mask_all_samples]
        dist_sample_to_map_points, nearest_map_points_ind = self.map_kd_tree.query(sample, k = self.max_num_map_points, distance_upper_bound=self.knn_radius)
        
        #some samples have less than max_num_map_points nearby map points
        mask_map_points = dist_sample_to_map_points < self.knn_radius
        #ind_dummpy_map_point = nearest_map_points_ind.max() 
       # nearest_map_points_ind = torch.from_numpy(nearest_map_points_ind.astype(int)).long()

        
        nearest_map_points_ind = nearest_map_points_ind[mask_map_points]

        unique_map_points_ind, unique_map_points_ind_mapping = np.unique(nearest_map_points_ind, return_index=True)
        #the node id in DGL graphs should start from zeros, otherwise it creates dummpy nodes.
        unique_map_points_ind_normalized = torch.arange(unique_map_points_ind.shape[0])
        map_ind_mapping = dict(zip(unique_map_points_ind, unique_map_points_ind_normalized))
        nearest_map_points_ind_normalized = np.vectorize(map_ind_mapping.get)(nearest_map_points_ind)

        unique_ind_sample = torch.arange(sample.shape[0]).unsqueeze(1).repeat(1, self.max_num_map_points)[mask_map_points]

        
        import pdb; pdb.set_trace()

        #record the pixel inds corresponding to each sample
        ind_pixels = np.arange(N0)[:, np.newaxis].repeat(M0, axis=1).flatten()[mask_all_samples]
        unique_ind_pixels, unique_ind_pixels_mapping = np.unique(ind_pixels, return_index=True)
        unique_ind_pixels_normalized = np.arange(unique_ind_pixels.shape[0])
        pixel_ind_mapping = dict(zip(unique_ind_pixels, unique_ind_pixels_normalized))
        ind_pixel_normalized = np.vectorize(pixel_ind_mapping.get)(ind_pixels)


        #put everything into our dgl graph format and build the graph
        g_data = {('voxel', 'neighbor', 'sample'):
                  (nearest_map_points_ind_normalized, unique_ind_sample),
                  ('sample', 'ray', 'pixel'):
                  (unique_ind_sample, ind_pixel_normalized),
                  }


        g = dgl.heterograph(g_data)

        #store data on the graph nodes
        g.nodes['sample'].data['dist_to_cam'] = steps0.repeat(N0,1).flatten()[mask_all_samples]
        g.nodes['sample'].data['sample_xyz'] = torch.from_numpy(sample).float()
        g.nodes['voxel'].data['ind_voxel'] = torch.from_numpy(unique_map_points_ind).float()
        g.nodes['pixel'].data['ind_pixel'] = torch.from_numpy(unique_ind_pixels).float()
        

        return g




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


