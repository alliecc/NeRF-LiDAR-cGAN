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
from src.utils import get_ray_dir, modify_K_resize, read_img
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

            #adjust K according to image size
            img, img_raw_size = read_img(os.path.join(self.cfg['path_data_folder'], self.data_folder_name, self.data['path_img']
                                    [ind]), self.img_size, self.data['K'][ind], return_raw_size=True)

            resize_ratio = min(img_raw_size[0], img_raw_size[1])/self.img_size

            K = modify_K_resize(self.data['K'][ind], resize_ratio, img_raw_size)


            graph = self.generate_dgl_graph_from_E(E, K)
            dgl.data.utils.save_graphs(path_graph, graph)

        else:
            graph = dgl.load_graphs(path_graph)[0][0]


    def generate_dgl_graph_from_E(self, E, K):
        u, v = torch.meshgrid(torch.arange(self.img_size), torch.arange(self.img_size))
        uv = torch.stack((u, v)).reshape(2, -1).transpose(0, 1)
        ray_dirs = get_ray_dir(uv, K, E)

        cam_center = torch.inverse(E)[0:3, 3]

        g = self.point_sampler_build_dgl_graph(cam_center, ray_dirs, E, K)



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
        nearest_map_points_ind = nearest_map_points_ind[mask_map_points]

        #the node id in DGL graphs should start from zeros, otherwise it creates dummpy nodes.
        unique_map_points_ind = np.unique(nearest_map_points_ind)
        
        unique_map_points_ind_normalized = torch.arange(unique_map_points_ind.shape[0])
        map_ind_mapping = dict(zip(unique_map_points_ind, unique_map_points_ind_normalized))
        nearest_map_points_ind_normalized = np.vectorize(map_ind_mapping.get)(nearest_map_points_ind)
        unique_ind_sample_to_map = torch.arange(sample.shape[0]).unsqueeze(1).repeat(1, self.max_num_map_points)[mask_map_points]

        #record the pixel inds corresponding to each sample
        ind_pixels = np.arange(N0)[:, np.newaxis].repeat(M0, axis=1).flatten()[mask_all_samples]
        unique_ind_pixels = np.unique(ind_pixels)
        unique_ind_pixels_normalized = np.arange(unique_ind_pixels.shape[0])
        pixel_ind_mapping = dict(zip(unique_ind_pixels, unique_ind_pixels_normalized))
        ind_pixel_normalized = np.vectorize(pixel_ind_mapping.get)(ind_pixels)

        unique_ind_sample_to_pixel = torch.arange(sample.shape[0])

        #put everything into our dgl graph format and build the graph
        g_data = {('voxel', 'neighbor', 'sample'):
                  (nearest_map_points_ind_normalized, unique_ind_sample_to_map),
                  ('sample', 'ray', 'pixel'):
                  (unique_ind_sample_to_pixel, ind_pixel_normalized),
                  }

        g = dgl.heterograph(g_data)
        print(g)

        #store data on the graph nodes
        g.nodes['sample'].data['opacity_from_lidar'] = torch.from_numpy(dist_sample_to_map_points[:,0]) < 0.05
        g.nodes['sample'].data['dist_to_cam'] = steps0.repeat(N0,1).flatten()[mask_all_samples]
        g.nodes['sample'].data['sample_xyz'] = torch.from_numpy(sample).float()
        g.nodes['voxel'].data['ind_voxel'] = torch.from_numpy(unique_map_points_ind.astype(np.int32)).long()
        g.nodes['pixel'].data['ind_pixel'] = torch.from_numpy(unique_ind_pixels).long()
        

        self.map = torch.from_numpy(self.map).float()
        #import pyvista as pv
        #pl = pv.Plotter()
        #pl.set_background('white')
        #cloud = pv.PolyData(self.map[unique_map_points_ind])
        #pl.add_mesh(cloud, color='k', render_points_as_spheres=True, point_size=1,  opacity=0.5)
        #cloud = pv.PolyData(sample)
        #pl.add_mesh(cloud, color='g', render_points_as_spheres=True, point_size=1,  opacity=0.5)
        #pl.show()
        #cloud = pv.PolyData(self.map)
        #pl.add_mesh(cloud, color='b', render_points_as_spheres=True, point_size=1,  opacity=0.5)
##
        #cloud = pv.PolyData(sample0_xyz[((sample0_xyz > map_min) * (sample0_xyz < map_max)).all(axis=1)])
        #pl.add_mesh(cloud, color='c', render_points_as_spheres=True, point_size=1,  opacity=0.5)

        #perform tight sampling

        self.perform_tight_sampling(g)

        #render LiDAR depth
        self.render_lidar_depth(g)

        return g


    def perform_tight_sampling(self, g):
        def func_msg(edges):
            return {'ind_voxel': edges.src['ind_voxel'],
                    'sample_xyz': edges.dst['sample_xyz'],
                    'dist_to_cam': edges.dst['dist_to_cam']}

        def func_reduce(nodes):
            sample_xyz = nodes.mailbox['sample_xyz']

            if sample_xyz.shape[1] == 1:
                return {'sample_select': torch.ones(sample_xyz.shape[0], dtype=torch.bool)}

            map_xyz = self.map[nodes.mailbox['ind_voxel']]

            diff = sample_xyz - map_xyz
            dot_product = (diff[:,0].unsqueeze(1) * diff[:,1:]).sum(axis=-1)
            mask_tight_sample = ~(dot_product>0).all(axis=1)


            #don't do tight sampling to the samples on the ground
            mask_sample_on_ground = self.map_ground_mask[nodes.mailbox['ind_voxel']].any(axis=1)
            mask_tight_sample[mask_sample_on_ground] = True


            return {'sample_select': mask_tight_sample}

        block = dgl.to_block(g, dst_nodes={'sample': torch.arange(g.number_of_nodes('sample'))})
        funcs = {}
        funcs[('voxel', 'neighbor', 'sample')] = (func_msg, func_reduce)
        block.multi_update_all(funcs, 'sum')

        g.remove_nodes((~block.dstdata['sample_select']['sample']).nonzero()[:, 0], 'sample')
        g.remove_nodes((g[('sample', 'ray', 'pixel')].in_degrees() == 0).nonzero()[:, 0], 'pixel')
        g.remove_nodes((g[('voxel', 'neighbor', 'sample')].out_degrees() == 0).nonzero()[:, 0], 'voxel')

        return g
        


    def render_lidar_depth(self, g):
        def func_msg(edges):
            return {'dist_to_cam': edges.src['dist_to_cam'], 'opacity_from_lidar':edges.src['opacity_from_lidar']}


        def func_reduce(nodes):
            
            dist_to_cam = self.depth_min + self.depth_max * nodes.mailbox['dist_to_cam']/self.ray_sample_steps
            opacity = nodes.mailbox['opacity_from_lidar'].float()

            ray_step_length = (self.depth_max-self.depth_min)/(self.ray_sample_steps-1)
#
            # if close to a map point, density is 1 otherwise 0 1 - torch.exp(-density * ray_step_length)
            
            acc_transmission = torch.cumprod(torch.cat([torch.ones(opacity.shape[0], 1), 1. - opacity + 1e-10], dim=1), -1)[:, :-1]
#
            blend_weight = opacity * acc_transmission

            acc = blend_weight.sum(axis=1)
            depth = torch.sum(dist_to_cam * blend_weight, dim=1) / (acc+eps)
            
            return {'depth_lidar': depth}

        funcs = {}
        block = dgl.to_block(g, dst_nodes={'pixel': torch.arange(g.number_of_nodes('pixel'))})
        funcs['sample', 'ray', 'pixel'] = (func_msg, func_reduce)
        block.multi_update_all(funcs, 'sum')
        g.nodes['pixel'].data['depth_lidar'] = block.dstdata['depth_lidar']['pixel'][:, None]


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


