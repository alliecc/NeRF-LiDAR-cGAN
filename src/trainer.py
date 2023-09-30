
import torch
import os
import lpips
import numpy as np
import open3d as o3d
from src.dataloader import get_data_loader
from torch.utils.tensorboard import SummaryWriter
from src.networks import LiDARNeRF
from .utils import depth_inv_to_color
from skimage.metrics import structural_similarity

loss_fn_alex = lpips.LPIPS(net='alex')


def get_exp_name_model(cfg):

    exp_name_model = f"{cfg['num_mlp_feat_layers']}_"
    exp_name_model += f"{cfg['num_mlp_alpha_layers']}_"
    exp_name_model += f"{cfg['num_mlp_rgb_layers']}_"
    exp_name_model += f"{cfg['voxel_feature_dim']}_"
    exp_name_model += f"{cfg['mlp_feat_dim']}_"
    exp_name_model += f"{cfg['nerf_rgb_loss_weight']}_"
    exp_name_model += f"{cfg['nerf_depth_loss_weight']}_"
    exp_name_model += f"{cfg['gan_color_loss_weight']}_"
    exp_name_model += f"{cfg['gan_loss_weight']}_"
    exp_name_model += f"{cfg['depth_loss_error_range']}_"
    exp_name_model += f"{cfg['lr']}"

    return exp_name_model


def get_exp_name_data(cfg):

    return cfg['name_data']


def compute_metrics(rgb_est, rgb_gt):

    # assume input pixel value range with -1~1
    x1 = (rgb_gt.detach().cpu().numpy() + 1)/2
    x2 = (rgb_est.detach().cpu().numpy() + 1)/2

    mask = x1 != 0
    diff = (x1-x2)[mask]

    mse = (diff**2).sum()/mask.sum()
    psnr = -10*np.log10(mse)
    ssim = structural_similarity(x1, x2, channel_axis=-1, data_range=1)

    # normalize to -1~1 for lpips
    x1 = rgb_gt.detach().cpu().permute(2, 0, 1)
    x2 = rgb_est.detach().cpu().permute(2, 0, 1)
    lpips = loss_fn_alex(x1, x2).item()

    return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'mse': mse}


class Trainer:

    def __init__(self, cfg, eval_only=True):

        exp_name_data = get_exp_name_data(cfg)
        exp_name_model = get_exp_name_model(cfg)

        exp_name = f"{exp_name_data}_{exp_name_model}_{cfg['log_id'][:4]}"
        self.log_writer = SummaryWriter(os.path.join(cfg['path_log'], 'logs', exp_name))
        path_lidar_map = os.path.join(cfg['path_data_folder'], 'argoverse_2_maps', f"{cfg['log_id']}_{cfg['name_data']}.ply")

        print('Load lidar map from', path_lidar_map)
        map_lidar = np.asarray(o3d.io.read_point_cloud(path_lidar_map).points)

        self.net = LiDARNeRF(cfg, map_lidar).to(cfg['device'])
        e_start = self.net.load_weights(cfg['path_weights'], exp_name, pretrained_path=cfg['path_pretrained_weight'])

        self.e_start = e_start
        self.e_end = int(cfg['num_epoch'])
        self.val_interval = int(cfg['val_interval'])
        self.iter_log_interval = int(cfg['iter_log_interval'])
        self.path_weights = cfg['path_weights']
        self.device = cfg['device']

        self.dataloader_train = get_data_loader(cfg, exp_name_data, 'train')
        self.dataloader_val = get_data_loader(cfg, exp_name_data, 'val')
        self.exp_name = exp_name

        # for evaluation

        self.list_metric = ['psnr', 'ssim', 'lpips', 'mse']
        self.num_iter = 0
        self.eval_only = eval_only

    def run(self):

        with torch.no_grad():
            self.run_val(self.dataloader_val, self.e_start-1)

        if not self.eval_only:
            for e in range(self.e_start, self.e_end):
                print('epoch', e)
                self.num_iter = e * len(self.dataloader_train)
                if e % self.val_interval == 0 and e != self.e_start:
                    with torch.no_grad():
                        self.run_val(self.dataloader_val, e)

                self.run_train(self.dataloader_train)

    def run_train(self, dataloader):
        print('train')
        for i, batch in enumerate(dataloader):
            # only support batch size == 1
            assert len(batch) == 1
            data_dict = batch[0]
            for key in data_dict:
                data_dict[key] = data_dict[key].to(self.device)

            self.num_iter += 1
            output = self.net(data_dict, training=True)

            if self.num_iter % self.iter_log_interval == 0:
                for k in output['dict_loss'].keys():
                    self.log_writer.add_scalar(f'train/{k}', output['dict_loss'][k], self.num_iter)

    def run_val(self, dataloader, e_val):
        print('val')
        dict_metrics_1, dict_metrics_2 = {}, {}
        for m in self.list_metric:
            dict_metrics_1[m] = []
            dict_metrics_2[m] = []

        for i, batch in enumerate(dataloader):

            # only support batch size == 1
            assert len(batch) == 1
            data_dict = batch[0]
            for key in data_dict:
                data_dict[key] = data_dict[key].to(self.device)

            output = self.net(data_dict)
            eval_1 = compute_metrics(output['img_rgb_gt'], output['img_rgb_nerf'])
            eval_2 = compute_metrics(output['img_rgb_gt'], output['img_rgb_gan'])

            for m in self.list_metric:
                dict_metrics_1[m].append(eval_1[m])
                dict_metrics_2[m].append(eval_2[m])

            if (i+1) % 5 == 0:  # just to visualize some val results
                for key in output:
                    if 'depth' in key:  # show inverse depth map
                        depth_inv = output[key].cpu().numpy()[:, :, 0]
                        depth_color = depth_inv_to_color(depth_inv)
                        self.log_writer.add_image(f'val_img_{key}/{i}', torch.from_numpy(depth_color).permute(2, 0, 1), self.num_iter)
                    elif 'rgb' in key:
                        mask = output[key].sum(axis=2) == 0
                        img = (output[key] + 1)/2
                        img[mask] = 0
                        self.log_writer.add_image(f'val_img_{key}/{i}', img.permute(2, 0, 1), self.num_iter)

        for m in self.list_metric:
            print(m, f'{np.mean(dict_metrics_1[m]):.4f}',
                  f'{np.std(dict_metrics_1[m]):.4f}',
                  f'{np.mean(dict_metrics_2[m]):.4f}',
                  f'{np.std(dict_metrics_2[m]):.4f}')

            self.log_writer.add_scalar(f'{m}/mean_stage_1', np.mean(dict_metrics_1[m]), e_val)
            self.log_writer.add_scalar(f'{m}/std_stage_1', np.std(dict_metrics_1[m]), e_val)
            self.log_writer.add_scalar(f'{m}/mean_stage_2', np.mean(dict_metrics_2[m]), e_val)
            self.log_writer.add_scalar(f'{m}/std_stage_2', np.std(dict_metrics_2[m]), e_val)

        self.net.save_weights(e_val, self.exp_name, self.path_weights, np.mean(dict_metrics_2['psnr']))
