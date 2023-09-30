import torch
import dgl
import os
import pickle
import open3d as o3d
from .utils import read_img

eps = 1e-10


def get_data_loader(cfg, name_exp, split):

    dataset = LiDARDataset(cfg, split, name_exp)
    if split == 'train':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


def collate_fn(list_x):
    return list_x


class LiDARDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, name_data):
        self.dataset_type = cfg['dataset_type']
        self.split = split
        self.name_data = name_data
        self.cfg = cfg
        self.img_size = 256

        self.data_folder_name = cfg['data_folder_name']

        path_dataset = os.path.join(cfg['path_data_folder'], self.data_folder_name, cfg['log_id'])

        if split == 'train':
            with open(os.path.join(path_dataset, f"{cfg['log_id']}_30_0.05_3_cams_{split}_colmap.pkl"), 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(os.path.join(path_dataset, f"{cfg['log_id']}_30_0.05_3_cams_{split}.pkl"), 'rb') as f:
                self.data = pickle.load(f)

        num_samples = len(self.data['path_img'])

        print(f"Load {cfg['log_id']} {split} dataset with {num_samples} samples.")

    def __len__(self):
        return len(self.data['path_img'])

    def __getitem__(self, ind):

        img = read_img(os.path.join(self.cfg['path_data_folder'], self.data_folder_name, self.data['path_img']
                                    [ind]), self.img_size, self.data['K'][ind])

        path_dyn_mask = os.path.join(self.cfg['path_dyn_masks'], self.cfg['log_id'], f'{self.split}_mask_{ind:04d}.pt')
        dyn_mask = torch.load(path_dyn_mask)
        path_graph = os.path.join(self.cfg['path_preprocessed_graph'], self.name_data, self.cfg['log_id'], f'{self.split}_{ind:04d}.bin')

        graph = dgl.load_graphs(path_graph)[0][0]
        cam_center = torch.from_numpy(self.data['E'][ind]).float().inverse()[0:3, 3]

        return {'graph': graph, 'img_rgb_gt': torch.from_numpy(img).float(), 'img_dyn_mask': dyn_mask, 'cam_center': cam_center}
