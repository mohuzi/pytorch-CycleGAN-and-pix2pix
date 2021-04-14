import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from torch.autograd import Variable
from . import networks
import math

class AugCycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(
            no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A',
                           'cycle_A',   'D_B', 'G_B', 'cycle_B','G_z_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A','fake_B_e1','fake_B_e2']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        # TODO
        if self.isTrain:
            self.model_names = ['G_A_B', 'G_B_A', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A_B', 'G_B_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A_B = networks.define_stochastic_G(opt.nlatent, opt.input_nc, opt.output_nc, opt.ngf,  opt.norm,
                                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B_A = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        enc_input_nc = opt.output_nc
        if opt.enc_A_B:
            enc_input_nc += opt.input_nc

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_z_B = networks.define_LAT_D(nlatent=opt.nlatent, ndf=opt.ndf,
                                                  use_sigmoid=opt.use_sigmoid,
                                                  init_type=opt.init_type,
                                                  init_gain=opt.init_gain,
                                                  gpu_ids=opt.gpu_ids)
        
        self.netE_B = networks.define_E(nlatent=opt.nlatent, input_nc=enc_input_nc,
                                nef=opt.nef, norm='batch',
                                init_type=opt.init_type,
                                init_gain=opt.init_gain,
                                gpu_ids=opt.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss() 
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.

            self.optimizer_G_A = torch.optim.Adam(self.netG_B_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.Adam(itertools.chain(self.netG_A_B.parameters(),
                                                                self.netE_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr/5., betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(),
                                                                self.netD_z_B.parameters(),
                                                                ),
                                                lr=opt.lr/5., betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""

        prior_z_B_e1 = Variable(self.real_A.data.new(self.real_A.size(0), self.opt.nlatent, 1, 1).normal_(0, 1))
        self.fake_B_e1 = self.netG_A_B.forward(self.real_A, prior_z_B_e1)

        prior_z_B_e2 = Variable(self.real_A.data.new(self.real_A.size(0), self.opt.nlatent, 1, 1).normal_(0, 1))
        self.fake_B_e2 = self.netG_A_B.forward(self.real_A, prior_z_B_e2)
 
    def forward(self,prior_z_B):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        self.prior_z_B = prior_z_B

        ##### genearte B using A and z_B : A ==> B <== z_B
        self.fake_B = self.netG_A_B.forward(self.real_A, prior_z_B)
        ##### generate A and z_B from B : A <-- B --> z_B
        self.fake_A = self.netG_B_A.forward(self.real_B) 
 
        if self.opt.enc_A_B:
            concat_B_A = torch.cat((self.fake_A, self.real_B), 1)
            self.mu_z_realB, self.logvar_z_realB = self.netE_B.forward(concat_B_A)
        else:
            self.mu_z_realB, self.logvar_z_realB = self.netE_B.forward(self.real_B)

        if self.opt.stoch_enc:
            self.post_z_realB = gauss_reparametrize(self.mu_z_realB, self.logvar_z_realB)
        else:
            self.post_z_realB = self.mu_z_realB.view(self.mu_z_realB.size(0), self.mu_z_realB.size(1), 1, 1)
            self.logvar_z_realB = self.logvar_z_realB * 0.0


        ##### A -> B -> A cycle loss
        self.rec_A = self.netG_B_A(self.fake_B)    

        ##### B -> A -> B cycle loss
        self.rec_B = self.netG_A_B(self.fake_A,self.post_z_realB)   

        # reconstruct z_B from A and fake_B : A ==> z_B <== fake_B
        if self.isTrain:
            if self.opt.enc_A_B:
                concat_A_B = torch.cat((self.real_A, self.fake_B), 1)
                self.mu_z_fakeB, self.logvar_z_fakeB = self.netE_B.forward(concat_A_B)
            else:
                self.mu_z_fakeB, self.logvar_z_fakeB = self.netE_B.forward(self.fake_B)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        #print(real.shape) #torch.Size([1, 16, 1, 1]) / ([1, 3, 256, 256])
        #print(fake.shape) #torch.Size([1, 1296, 1, 1]) / ([1, 3, 256, 256])
        
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def backward_D_z_B(self):
        """Calculate GAN loss for discriminator D_B""" 
        self.loss_D_B = self.backward_D_basic(self.netD_z_B, self.prior_z_B, self.post_z_realB )

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
 

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        self.loss_G_z_B = self.criterionGAN(self.netD_z_B(self.post_z_realB), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(
            self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(
            self.rec_B, self.real_B) * lambda_B

        # minimize the NLL of original z_B sample
        bs = self.prior_z_B.size(0)
        if self.opt.stoch_enc:
            log_prob_z_B = log_prob_gaussian(self.prior_z_B.view(bs, self.opt.nlatent),
                                             self.mu_z_fakeB.view(bs, self.opt.nlatent),
                                             self.logvar_z_fakeB.view(bs, self.opt.nlatent))

            loss_cycle_z_B = -1.0 * log_prob_z_B.mean(1).mean(0)* self.opt.lambda_z_B
        else:
            loss_cycle_z_B = self.criterionCycle(self.mu_z_fakeB.view(bs, self.opt.nlatent),
                                       self.prior_z_B.view(bs, self.opt.nlatent))* self.opt.lambda_z_B

        # measure KLD
        kld_z_B = kld_std_guss(self.mu_z_realB, self.logvar_z_realB).mean(0)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + \
            self.loss_cycle_B  + loss_cycle_z_B

        if self.opt.stoch_enc:
            self.loss_G += kld_z_B * self.opt.lambda_z_B

        if self.opt.z_gan and not self.opt.stoch_enc:
            self.loss_G += self.loss_G_z_B


        self.loss_G.backward()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            prior_z_B = Variable(self.real_A.data.new(self.real_A.size(0), self.opt.nlatent, 1, 1).normal_(0, 1))
            self.forward(prior_z_B)
            self.compute_visuals()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        prior_z_B = Variable(self.real_A.data.new(self.real_A.size(0), self.opt.nlatent, 1, 1).normal_(0, 1))
        self.forward(prior_z_B)      # compute fake images and reconstruction images.

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
        self.optimizer_D_B.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        if self.opt.z_gan and not self.opt.stoch_enc:
            self.backward_D_z_B()
        self.optimizer_D_A.step()  # update D_A and D_B's weights
        self.optimizer_D_B.step()  # update D_A and D_B's weights

        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G_A.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_G_B.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G_A.step()       # update G_A and G_B's weights
        self.optimizer_G_B.step()       # update G_A and G_B's weights
 
def gauss_reparametrize(mu, logvar, n_sample=1):
    """Gaussian reparametrization"""
    std = logvar.mul(0.5).exp_()
    size = std.size()
    eps = Variable(std.data.new(size[0], n_sample, size[1]).normal_())
    z = eps.mul(std[:, None, :]).add_(mu[:, None, :])
    z = torch.clamp(z, -4., 4.)
    return z.view(z.size(0)*z.size(1), z.size(2), 1, 1)

def log_prob_gaussian(z, mu, log_var):
    res = - 0.5 * log_var - ((z - mu)**2.0 / (2.0 * torch.exp(log_var)))
    res = res - 0.5 * math.log(2*math.pi)
    return res


def kld_std_guss(mu, log_var):
    """
    from Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kld = -0.5 * torch.sum(log_var + 1. - mu**2 - torch.exp(log_var), dim=1)
    return kld