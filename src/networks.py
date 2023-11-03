import torch
import os
import cv2
import glob
import dgl
import pix2pix.models as gan_models
import numpy as np
from pix2pix.options.train_options import TrainOptions as pix2pix_options
from .utils import positional_encoding, clip_values
eps = 1e-10


class LiDARNeRF(torch.nn.Module):
    # part of the MLP code was inherited from PointNeRF codebase

    def __init__(self, cfg, map_xyz):
        super(LiDARNeRF, self).__init__()

        self.F_m = int(cfg['voxel_feature_dim'])
        self.F_h = int(cfg['mlp_feat_dim'])
        self.depth_max = float(cfg['depth_max'])
        self.depth_min = float(cfg['depth_min'])
        self.depth_loss_error_range = float(cfg['depth_loss_error_range'])
        self.ray_sample_steps = float(cfg['ray_sample_steps'])
        self.nerf_depth_loss_weight = float(cfg['nerf_depth_loss_weight'])
        self.nerf_rgb_loss_weight = float(cfg['nerf_rgb_loss_weight'])
        self.gan_color_loss_weight = float(cfg['gan_color_loss_weight'])

        self.device = cfg['device']

        # initialize LiDAR embeddings
        self.map_xyz = torch.from_numpy(map_xyz).float().to(self.device)
        self.map_feat = torch.nn.parameter.Parameter(torch.FloatTensor(map_xyz.shape[0], self.F_m), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.map_feat)

        self.sigmoid = torch.nn.Sigmoid()
        self.l2loss = torch.nn.MSELoss()
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.1)

        self.num_mlp_feat_layers = int(cfg['num_mlp_feat_layers'])
        self.num_mlp_alpha_layers = int(cfg['num_mlp_alpha_layers'])
        self.num_mlp_rgb_layers = int(cfg['num_mlp_rgb_layers'])

        self.F_xyz = 3
        self.freq_PE = 4
        self.depth_min = 1
        self.img_size = 256
        self.gan_dilate_kernel = 5

        # initialize the MLPs
        self.mlp_init()

        # initialize the cGAN
        opt = pix2pix_options().parse()
        if cfg['device'] == 'cpu':
            opt.gpu_ids = []

        self.pix2pix = gan_models.pix2pix_model.Pix2PixModel(opt)
        self.pix2pix.gan_loss_weight = float(cfg['gan_loss_weight'])
        self.pix2pix.optimizer_G = torch.optim.Adam(list(self.pix2pix.netG.parameters()) +
                                                    list(self.parameters()), lr=float(cfg['lr']))  
        self.pix2pix.use_mask = True

    def mlp_init(self):

        in_channels = self.F_m + self.F_xyz * self.freq_PE * 2

        out_channels = self.F_h
        mlp_feat = []
        for i in range(self.num_mlp_feat_layers):
            mlp_feat.append(torch.nn.Linear(in_channels, out_channels))
            mlp_feat.append(self.leakyrelu)
            in_channels = out_channels

        self.mlp_feat = torch.nn.Sequential(*mlp_feat)

        alpha_block = []
        in_channels = self.F_h
        out_channels = int(self.F_h / 2)

        for i in range(self.num_mlp_alpha_layers - 1):
            alpha_block.append(torch.nn.Linear(in_channels, out_channels))
            alpha_block.append(self.leakyrelu)
            in_channels = out_channels

        alpha_block.append(torch.nn.Linear(in_channels, 1))

        # predict opacity, not density
        alpha_block.append(self.sigmoid)

        self.alpha_branch = torch.nn.Sequential(*alpha_block)

        rgb_block = []
        in_channels = self.F_h
        out_channels = int(self.F_h / 2)
        for i in range(self.num_mlp_rgb_layers - 1):
            rgb_block.append(torch.nn.Linear(in_channels, out_channels))
            rgb_block.append(self.leakyrelu)
            in_channels = out_channels

        rgb_block.append(torch.nn.Linear(in_channels, 3))
        rgb_block.append(self.sigmoid)

        self.rgb_branch = torch.nn.Sequential(*rgb_block)

        self.init_seq(self.mlp_feat)
        self.init_seq(self.alpha_branch)
        self.init_seq(self.rgb_branch)

    def init_seq(self, m):
        # initialize the MLP layers
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
            m.bias.data.fill_(0.01)

    def load_ckp(self, path_ckp):
        ckp = torch.load(path_ckp)
        self.load_state_dict(ckp['map_state_dict'])
        self.pix2pix.netG.load_state_dict(ckp['pix2pix_G_state_dict'])
        self.pix2pix.netD.load_state_dict(ckp['pix2pix_D_state_dict'])

    def load_weights(self, path_models, exp_name, pretrained_path=None):

        print('***********************************')
        if os.path.exists(pretrained_path):
            print('Start from an assigned pretrained model : ', pretrained_path)
            ckp = torch.load(pretrained_path)
            self.load_ckp(ckp)
            return 0

        path_ckp = os.path.join(path_models, f'{exp_name}')
        name_ckp = f'latest_*.ckp'
        list_ckp = glob.glob(os.path.join(path_ckp, name_ckp))

        if len(list_ckp) == 0:
            # if no pretrained_path or epoch is assigned, start from the existing trained weights or initialization.
            print('No pretrained model to start from', path_ckp)
            model_dict = self.state_dict()
            pretrained_name = os.path.join('init_weights',
                                           f"init_mlp_{self.num_mlp_feat_layers}_{self.num_mlp_rgb_layers}_{self.num_mlp_alpha_layers}_{self.F_h}_{self.F_m}.pt")

            if os.path.exists(pretrained_name):
                pretrained_dict = torch.load(pretrained_name)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)
                print(f'Start MLPs from default initial weights')
            else:
                print('No defailt initial weights are found!')
                print(f'Start from initialization')
            return 0

        else:
            assert len(list_ckp) == 1
            path_ckp = list_ckp[0]
            epoch = int(path_ckp.split('/')[-1].split('_')[1].split('.')[0])

        print('Load parameters from ', path_ckp)
        print(f'Start from epoch {epoch}')

        self.load_ckp(path_ckp)

        return epoch+1

    def get_state_dict(self, e):
        return {
            'epoch': e,
            'map_state_dict': self.state_dict(),
            'pix2pix_G_state_dict': self.pix2pix.netG.state_dict(),
            'pix2pix_D_state_dict': self.pix2pix.netD.state_dict()}

    def save_weights(self, e, exp_name, path_models, current_psnr=0):

        if not os.path.exists(path_models):
            os.mkdir(path_models)

        path_ckp = os.path.join(path_models, exp_name)
        if not os.path.exists(path_ckp):
            os.mkdir(path_ckp)

        print(f'Epoch {e}: save parameters to ', path_ckp)

        # map feat can be large, remove old ckps
        list_existing_ckp = glob.glob(os.path.join(path_ckp, f'latest_*.ckp'))

        if len(list_existing_ckp) == 1:
            previous_latest_epoch = int(list_existing_ckp[0].split('/')[-1].split('_')[1].split('.')[0])
        else:
            previous_latest_epoch = -1

        # only save when the current model is more trained
        if previous_latest_epoch <= e:
            if len(list_existing_ckp) == 1:
                os.remove(list_existing_ckp[0])

            torch.save(self.get_state_dict(e),
                       os.path.join(path_ckp, f'latest_{e}.ckp'))

    def build_img_depth(self, d, ind_pixel, inverse=True):

        out = torch.zeros(self.img_size, self.img_size, 1, device=self.device).reshape(-1, 1)
        out[ind_pixel] = d

        if inverse:
            out[out < self.depth_min] = 0
            out[out != 0] = 1 / out[out != 0]

        return out

    def dilate_ind_pixel(self, ind_pixel):

        img_dilated_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8).flatten()
        img_dilated_mask[ind_pixel.cpu().numpy()] = 255
        img_dilated_mask = img_dilated_mask.reshape((self.img_size, self.img_size))
        kernel = np.ones((self.gan_dilate_kernel, self.gan_dilate_kernel), np.uint8)
        img_dilated_mask = cv2.dilate(img_dilated_mask, kernel, iterations=1)
        ind_pixel_dilated = torch.from_numpy(img_dilated_mask.flatten() != 0).to(self.device)

        return ind_pixel_dilated

    def render_pixel(self, g):

        map_feat = self.map_feat[g.nodes['voxel'].data['ind_voxel']]
        g.nodes['voxel'].data['map_feat'] = map_feat

        # aggregate ray sample features from LiDAR points ('voxel's) and predict
        funcs = {}
        funcs[('voxel', 'neighbor', 'sample')] = (self.func_msg_voxel_to_sample,
                                                  self.func_reduce_voxel_to_sample)

        block_sample = dgl.to_block(g, dst_nodes={'sample': torch.arange(g.number_of_nodes('sample'))})
        block_sample.multi_update_all(funcs, 'sum')

        g.nodes['sample'].data['rgb'] = block_sample.dstdata['rgb']['sample']
        g.nodes['sample'].data['alpha'] = block_sample.dstdata['alpha']['sample']

        # render pixel colors
        funcs = {}
        funcs[('sample', 'ray', 'pixel')] = (self.func_msg_sample_to_pixel,
                                             self.func_reduce_sample_to_pixel)

        block_pixel = dgl.to_block(g, dst_nodes={'pixel': torch.arange(g.number_of_nodes('pixel'))})
        block_pixel.multi_update_all(funcs, 'sum')

        return block_pixel.dstdata['pixel_rgb']['pixel'], block_pixel.dstdata['pixel_d']['pixel'], block_pixel.dstdata['acc']['pixel']

    def func_msg_voxel_to_sample(self, edges):
        feat = edges.src['map_feat']
        voxel_xyz = self.map_xyz[edges.src['ind_voxel']]
        sample_xyz = edges.dst['sample_xyz']

        offset = sample_xyz - voxel_xyz

        feat = torch.cat([feat, positional_encoding(offset, self.freq_PE)], dim=-1)
        feat = self.mlp_feat(feat)
        weight = torch.exp(-10 * offset.norm(dim=1))

        return {'map_feat':  feat, 'weight': weight}

    def func_reduce_voxel_to_sample(self, nodes):

        feat = nodes.mailbox['map_feat']
        weight = nodes.mailbox['weight']
        weight = weight / (weight.sum(axis=1).unsqueeze(1))

        feat = (feat * weight.unsqueeze(2)).sum(axis=1)

        alpha = self.alpha_branch(feat)
        rgb = self.rgb_branch(feat)

        return {'alpha': alpha, 'rgb': rgb}

    def func_msg_sample_to_pixel(self, edges):

        return {'alpha': edges.src['alpha'],
                'rgb': edges.src['rgb'],
                'dist_to_cam': edges.src['dist_to_cam']}

    def func_reduce_sample_to_pixel(self, nodes):

        # we predict opacity directly
        opacity = nodes.mailbox['alpha']

        f = nodes.mailbox['rgb']
        dist = self.depth_min + self.depth_max * \
            nodes.mailbox['dist_to_cam']/self.ray_sample_steps

        acc_transmission = torch.cumprod(torch.cat([torch.ones(opacity.shape[0], 1, 1).to(
            opacity.device), 1. - opacity + eps], dim=1), -1)[:, :-1, :]
        blend_weight = opacity * acc_transmission
        pixel_feat = torch.sum(f * blend_weight, dim=1)

        acc = blend_weight.sum(axis=1)
        if dist.shape[1] == 1:
            pixel_depth = dist
        else:
            pixel_depth = torch.sum(dist.unsqueeze(2) * blend_weight, dim=1) / (eps + acc)

        return {'pixel_rgb': pixel_feat, 'pixel_d': pixel_depth,  'acc': acc}

    def forward(self, input_dict, training=False):

        dict_loss = {}
        g = input_dict['graph']
        depth_l = g.nodes['pixel'].data['depth_lidar']
        ind_pixel_valid = g.nodes['pixel'].data['ind_pixel']

        rgb, depth, _ = self.render_pixel(g)
        rgb = rgb * 2-1  # rescale from 0-1 to -1~1 for pix2pix
        img_rgb = torch.zeros(self.img_size, self.img_size, 3, device=self.device).reshape(-1, 3)
        img_rgb[ind_pixel_valid] = rgb

        img_depth_inv = self.build_img_depth(depth, ind_pixel_valid)
        img_depth_l_inv = self.build_img_depth(depth_l, ind_pixel_valid)

        # we can use cGAN to fill some holes from point-based NeRF
        # this is done by dilating the mask a bit
        ind_pixel_dilate = self.dilate_ind_pixel(ind_pixel_valid)

        img_rgb_gt = torch.zeros_like(img_rgb)
        img_rgb_gt[ind_pixel_dilate] = input_dict['img_rgb_gt'].reshape(-1, 3)[ind_pixel_dilate]

        # for pix2pix input
        img_input = torch.cat([img_rgb, img_depth_inv], axis=1).reshape(self.img_size, self.img_size, 4)
        img_target = torch.cat([img_rgb_gt, img_depth_inv], axis=1).reshape(self.img_size, self.img_size, 4)

        self.pix2pix.set_input({'A': img_input.permute(2, 0, 1).unsqueeze(0),
                                'B': img_target.permute(2, 0, 1).unsqueeze(0),
                                'A_paths': '',
                                'B_paths': '',
                                })

        # remove dynamic obj from cGAN losses
        self.pix2pix.dyn_mask = input_dict['img_dyn_mask']

        # remove invalid pixels from cGAN output
        self.pix2pix.mask = torch.zeros(self.img_size, self.img_size, dtype=torch.bool,
                                        device=self.device).flatten()
        self.pix2pix.mask[ind_pixel_dilate] = True
        self.pix2pix.mask = self.pix2pix.mask.reshape(self.img_size, self.img_size)

        # run cGAN
        self.pix2pix.forward(training)
        img_rgb_gan = self.pix2pix.fake_B[0][0:3].permute(1, 2, 0)

        mask_d_loss = (depth_l > self.depth_min) * (torch.abs(depth_l-depth) > self.depth_loss_error_range)
        depth_error = torch.zeros_like(depth)
        depth_error[mask_d_loss] = torch.abs(1/(depth[mask_d_loss]+eps)-1/depth_l[mask_d_loss])
        img_depth_error = self.build_img_depth(depth_error, ind_pixel_valid, inverse=False)

        # if training, compute the losses and do backward
        if training:
            # add losses to the generator loss in cGAN
            self.pix2pix.loss_G = 0

            # compute depth loss
            loss_d = depth_error.mean()  
            dict_loss['loss_lidar_depth'] = loss_d
            self.pix2pix.loss_G += self.nerf_depth_loss_weight * loss_d

            # compute stage 1 rgb loss, remove dynamic objs
            dyn_mask = input_dict['img_dyn_mask'].flatten()[ind_pixel_valid]
            loss_rgb_1 = self.nerf_rgb_loss_weight * self.l2loss(rgb[~dyn_mask],
                                                                 img_rgb_gt[ind_pixel_valid][~dyn_mask])
            dict_loss['loss_rgb_1'] = loss_rgb_1
            self.pix2pix.loss_G += loss_rgb_1

            # compute stage 2 rgb loss, remove dynamic objs
            dyn_mask_dilate = input_dict['img_dyn_mask'].flatten()[ind_pixel_dilate]
            rgb_gan = img_rgb_gan.reshape(-1, 3)
            self.pix2pix.loss_G += self.gan_color_loss_weight * self.l2loss(rgb_gan[ind_pixel_dilate]
                                                                            [~dyn_mask_dilate], img_rgb_gt[ind_pixel_dilate][~dyn_mask_dilate])

            # gan_loss_weight is applied inside the model
            # do backward
            self.pix2pix.optimize_parameters(self.map_feat, do_forward=False)
            dict_loss['loss_G'] = self.pix2pix.loss_G
            dict_loss['loss_D'] = self.pix2pix.loss_D

        out = {'img_rgb_gan': clip_values(img_rgb_gan).reshape(self.img_size, self.img_size, 3),
               'img_depth_inv': img_depth_inv.reshape(self.img_size, self.img_size, 1),
               'img_depth_l_inv': img_depth_l_inv.reshape(self.img_size, self.img_size, 1),
               'img_depth_error': img_depth_error.reshape(self.img_size, self.img_size, 1),
               'img_rgb_nerf': clip_values(img_rgb).reshape(self.img_size, self.img_size, 3),
               'img_rgb_gt': img_rgb_gt.reshape(self.img_size, self.img_size, 3)}

        if training:
            out['dict_loss'] = dict_loss

        return out
