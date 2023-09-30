import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns: 
            the modified parser.  

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1 
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # modified: we train one frame in each batch, so removed the batchnorm 'batch' -> 'norm'
       # parser.set_defaults(norm='none', netG='unet_256', dataset_mode='aligned')
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(norm='none', dataset_mode='aligned')

        if is_train:
            #parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            #self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.sigmoid = torch.nn.Sigmoid()  # edit: use sigmoid on netG output because the way we normalize the color
        self.loss_D = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if input['B'] is None:  # only supports AtoB
            self.real_B = None
        else:
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, training=True):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        out = self.netG(self.real_A)  # G(A)
        if self.use_mask:
            self.fake_B = torch.zeros_like(out)
            self.fake_B[:, :, self.mask] = out[:, :, self.mask]
        else:
            self.fake_B = out

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        # only take the first 3 channels from B, the appended depth is the same

        fake_AB = torch.cat((self.real_A, self.fake_B[:, 0:3]), 1)
        #remove dynamic obj effect from gan loss
        fake_AB[:,:,self.dyn_mask] = 0
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if self.real_B is not None:
            real_AB = torch.cat((self.real_A, self.real_B[:, 0:3]), 1)
            real_AB[:,:,self.dyn_mask] = 0
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
        else:
            self.loss_D_real = 0
        # combine loss and calculate gradients
        self.loss_D = self.gan_loss_weight * (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B

        # do this outside to apply masks!
        # if self.real_B is not None:
#
        #    self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100 #self.opt.lambda_L1 #+ 20 * self.criterionL1(self.real_A[:,0:3], self.real_B)
        #
        # else:
        #    self.loss_G_L1 = 0
        # combine loss and calculate gradients
        self.loss_G += self.gan_loss_weight * self.loss_G_GAN  # + self.loss_G_L1
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self, map_feat, do_forward=True):
        if do_forward:
            self.forward()                   # compute fake images: G(A)
        # update D
        if self.gan_loss_weight != 0:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero

            # torch.cuda.empty_cache()
            self.backward_D()                # calculate gradients for D

            torch.nn.utils.clip_grad_norm_(self.netD.module.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(map_feat, 1)

            self.optimizer_D.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        # torch.cuda.empty_cache()
        self.backward_G()                   # calculate graidents for G
        # torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(self.netG.module.parameters(), 1)
        #torch.nn.utils.clip_grad_norm_(map_feat, 10)

        self.optimizer_G.step()             # udpate G's weights
