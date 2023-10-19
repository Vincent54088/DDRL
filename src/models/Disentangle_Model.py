import torch
import torch.nn as nn
import torch.nn.functional as F

from models import network
from models.Mutual_Info import CLUB
from utils.perceptual import PerceptualLoss
from pytorch_msssim import ssim

class DisentangleNet(nn.Module):
    def __init__(self,opts):
        super(DisentangleNet,self).__init__()
        if opts.phase == 'train':
            lr = opts.lr
        else:
            lr = 0.0
        lr_Dcontent = lr/2.
        self.batch = opts.batch_size//2
        self.nz = opts.nz
        self.tch = opts.tch
        self.concat = opts.concat
        self.no_ms = opts.no_ms
        self.img_size = 256
        map_size = self.img_size // 4
        

        # image generator discriminators(two type : one is latent vector generate image,another is noise generate image)
        # multi scale disc
        self.discA = network.MultiScaleDis(3,opts.disc_scale,norm = opts.disc_norm,sn=opts.disc_spectral_norm)
        self.discB = network.MultiScaleDis(3,opts.disc_scale,norm = opts.disc_norm,sn=opts.disc_spectral_norm)
        self.discA2 = network.MultiScaleDis(3,opts.disc_scale,norm = opts.disc_norm,sn=opts.disc_spectral_norm)
        self.discB2 = network.MultiScaleDis(3,opts.disc_scale,norm = opts.disc_norm,sn=opts.disc_spectral_norm)

        self.discContent = network.Dis_content(out_tch=self.tch*4,map_size=map_size)
        self.MI_estimation = CLUB(X_dim=self.tch*4,Y_dim=opts.nz,hidden_size=1024,map_size=map_size)

        # style and content Encoders
        self.enc_c = network.E_content(input_dim_a=3,input_dim_b=3,tch=self.tch)
        if self.concat:
            self.enc_s = network.E_attr_concat(3,3,self.nz,norm_layer=None,nl_layer=network.get_non_linearity(layer_type='lrelu'))
        
        # generator
        if self.concat:
            self.gen = network.G_concat(3,3,self.nz,tch=self.tch*4)
        
        # optimizers
        self.discA_optimizer = torch.optim.Adam(self.discA.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.discB_optimizer = torch.optim.Adam(self.discB.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.discA2_optimizer = torch.optim.Adam(self.discA2.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.discB2_optimizer = torch.optim.Adam(self.discB2.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.discContent_optimizer = torch.optim.Adam(self.discContent.parameters(),lr=lr_Dcontent,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.enc_c_optimizer = torch.optim.Adam(self.enc_c.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.enc_s_optimizer = torch.optim.Adam(self.enc_s.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        self.MI_estimation_optimizer = torch.optim.Adam(self.MI_estimation.parameters(),lr=0.0001,betas=(0.5,0.999),weight_decay=0.0001,eps=1e-4)
        #loss function
        loss_func = nn.MSELoss()
        layer_indexs = [3, 8, 15, 22]
        device = torch.device("cuda:%s" %str(opts.gpu) if torch.cuda.is_available() else "cpu")
        self.criterionL1 = PerceptualLoss('./PTH/vgg/vgg19.pth',loss_func, layer_indexs, device)
        self.flag = True

    def initialize(self):
        self.discA.apply(network.gaussian_weights_init)
        self.discB.apply(network.gaussian_weights_init)
        self.discA2.apply(network.gaussian_weights_init)
        self.discB2.apply(network.gaussian_weights_init)
        self.discContent.apply(network.gaussian_weights_init)
        # self.enc_c.apply(network.gaussian_weights_init)
        self.enc_s.apply(network.gaussian_weights_init)
        # self.gen.apply(network.gaussian_weights_init)
        self.MI_estimation.apply(network.gaussian_weights_init)

    def set_scheduler(self,opts,last_ep = 0):
        self.discA_scheduler = network.get_scheduler(self.discA_optimizer,opts,last_ep)
        self.discB_scheduler = network.get_scheduler(self.discB_optimizer,opts,last_ep)
        self.discA2_scheduler = network.get_scheduler(self.discA2_optimizer,opts,last_ep)
        self.discB2_scheduler = network.get_scheduler(self.discB2_optimizer,opts,last_ep)
        self.discContent_scheduler = network.get_scheduler(self.discContent_optimizer,opts,last_ep)
        self.enc_c_scheduler = network.get_scheduler(self.enc_c_optimizer,opts,last_ep)
        self.enc_s_scheduler = network.get_scheduler(self.enc_s_optimizer,opts,last_ep)
        self.gen_scheduler = network.get_scheduler(self.gen_optimizer,opts,last_ep)
        self.MI_estimation_scheduler = network.get_scheduler(self.MI_estimation_optimizer,opts,last_ep)
    
    def update_lr(self):
        self.discA_scheduler.step()
        self.discB_scheduler.step()
        self.discA2_scheduler.step()
        self.discB2_scheduler.step()
        self.discContent_scheduler.step()
        self.enc_c_scheduler.step()
        self.enc_s_scheduler.step()
        self.gen_scheduler.step()
        self.MI_estimation_scheduler.step()
    
    # reparemeterize
    def reparemeterize(self,mu,logvar,eps = None):
        std = torch.exp(0.5*logvar)
        if eps is None:
            eps = torch.randn(std.size(0),std.size(1)).cuda(self.gpu)
        return mu + std*eps
    
    def set_gpu(self,gpu):
        self.gpu = gpu
        self.discA.cuda(self.gpu)
        self.discB.cuda(self.gpu)
        self.discA2.cuda(self.gpu)
        self.discB2.cuda(self.gpu)
        self.discContent.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_s.cuda(self.gpu)
        self.gen.cuda(self.gpu)
        self.MI_estimation.cuda(self.gpu)

    def update_D_content(self,image_a,image_b,name):
        '''
            Discriminator make the content map of domain A (fog) to trend 0,which from domain B(clear) to trend 1
        '''
        
        half_size = self.batch
        self.real_A = image_a[0:half_size]
        self.real_B = image_b[0:half_size]
        self.z_content_a,self.z_content_b = self.enc_c.forward(self.real_A,self.real_B)#Size(1,256,64,64)
            
        # update content Disc
        self.discContent_optimizer.zero_grad()

        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.discContent.parameters(), 5)
        self.discContent_optimizer.step()
    
    def update_MI_estimator(self,image_a,image_b):
        self.z_content_a,self.z_content_b = self.enc_c.forward(image_a,image_b)
        # get encoded style Z
        if self.concat:
            self.mu_a,self.logvar_a,self.mu_b,self.logvar_b = self.enc_s.forward(image_a,image_b)
            self.z_style_a = self.reparemeterize(self.mu_a,self.logvar_a)
            self.z_style_b = self.reparemeterize(self.mu_b,self.logvar_b)
        else:
            self.z_style_a = self.enc_s.forward_a(image_a)
            self.z_style_b = self.enc_s.forward_b(image_b)
        
        self.z_content = torch.cat([self.z_content_a,self.z_content_b],dim = 0)
        self.z_style = torch.cat([self.z_style_a,self.z_style_b],dim = 0)

        # update MI upper bound estimator q(y|x)
        self.MI_estimation.train()
        MI = self.MI_estimation.learning_loss(self.z_content,self.z_style)
        self.MI_loss = MI.item()
        self.MI_estimation_optimizer.zero_grad()
        MI.backward(retain_graph = True)
        # nn.utils.clip_grad_norm_(self.MI_estimation.parameters(), 5)
       
        self.MI_estimation_optimizer.step()

        # update enc_c enc_s
        self.MI_estimation.eval()
        sampler_loss = self.MI_estimation.forward(self.z_content,self.z_style)
        self.sampler_loss = sampler_loss.item()

        self.enc_c_optimizer.zero_grad()
        self.enc_s_optimizer.zero_grad()
        
        sampler_loss.backward()
        self.enc_c_optimizer.step()
        self.enc_s_optimizer.step()

    def forward(self):
        half_size = self.batch
        real_A = self.input_A
        real_B = self.input_B
        self.real_A = real_A[0:half_size]
        self.real_A_random = real_A[half_size:]
        self.real_B = real_B[0:half_size]
        self.real_B_random = real_B[half_size:]

        #get encoded content map
        self.z_content_a,self.z_content_b = self.enc_c.forward(self.real_A,self.real_B)

        #get encoded style Z
        if self.concat:
            self.mu_a,self.logvar_a,self.mu_b,self.logvar_b = self.enc_s.forward(self.real_A,self.real_B)
            self.z_style_a = self.reparemeterize(self.mu_a,self.logvar_a)
            self.z_style_b = self.reparemeterize(self.mu_b,self.logvar_b)
        else:
            self.z_style_a = self.enc_s.forward_a(self.real_A)
            self.z_style_b = self.enc_s.forward_b(self.real_B)
        
        #get random z_s
        self.z_random = torch.randn(self.real_A.size(0),self.nz).cuda(self.gpu)
        if not self.no_ms:
            self.z_random2 = torch.randn(self.real_A.size(0),self.nz).cuda(self.gpu)
        
        #cross translation
        if not self.no_ms:
            # Size(4,96,64,64)
            input_content_forA = torch.cat((self.z_content_b,self.z_content_a,self.z_content_b,self.z_content_b),dim = 0)
            input_content_forB = torch.cat((self.z_content_a,self.z_content_b,self.z_content_a,self.z_content_a),dim = 0)
            # Size(4,8)
            input_style_forA = torch.cat((self.z_style_a,self.z_style_a,self.z_random,self.z_random2),dim = 0)
            input_style_forB = torch.cat((self.z_style_b,self.z_style_b,self.z_random,self.z_random2),dim = 0)
            output_fakeA = self.gen.forward_a(input_content_forA,input_style_forA)
            output_fakeB = self.gen.forward_b(input_content_forB,input_style_forB)

            assert torch.any(torch.isnan(output_fakeA)) == False or torch.any(torch.isnan(output_fakeB)),print('nan')

            self.fake_A_BA,self.fake_A_AA,self.fake_A_BRandom,self.fake_A_BRandom2 = torch.split(output_fakeA,self.z_content_a.size(0),dim = 0)
            self.fake_B_AB,self.fake_B_BB,self.fake_B_ARandom,self.fake_B_ARandom2 = torch.split(output_fakeB,self.z_content_a.size(0),dim = 0)
        else:
            # Size(3,256,64,64)
            input_content_forA = torch.cat((self.z_content_b,self.z_content_a,self.z_content_b),dim = 0)
            input_content_forB = torch.cat((self.z_content_a,self.z_content_b,self.z_content_a),dim = 0)
            # Size(3,8)
            input_style_forA = torch.cat((self.z_style_a,self.z_style_a,self.z_random),dim = 0)
            input_style_forB = torch.cat((self.z_style_b,self.z_style_b,self.z_random),dim = 0)
            output_fakeA = self.gen.forward_a(input_content_forA,input_style_forA)
            output_fakeB = self.gen.forward_b(input_content_forB,input_style_forB)
            self.fake_A_BA,self.fake_A_AA,self.fake_A_BRandom = torch.split(output_fakeA,self.z_content_a.size(0),dim = 0)
            self.fake_B_AB,self.fake_B_BB,self.fake_B_ARandom = torch.split(output_fakeB,self.z_content_a.size(0),dim = 0)
       
        
        #display
        self.image_display = torch.cat((self.real_A[0:1].detach().cpu(), self.fake_A_BA[0:1].detach().cpu(), \
                                    self.fake_A_BRandom[0:1].detach().cpu(), self.fake_A_AA[0:1].detach().cpu(), 
                                    self.real_B[0:1].detach().cpu(), self.fake_B_AB[0:1].detach().cpu(), \
                                    self.fake_B_ARandom[0:1].detach().cpu(), self.fake_B_BB[0:1].detach().cpu()), dim=0)

        # for latent regression
        if self.concat:
            self.mu2_a,_,self.mu2_b,_ = self.enc_s.forward(self.fake_A_BRandom,self.fake_B_ARandom)
        else:
            self.z_s_random_a,self.z_s_random_b = self.enc_s.forward(self.fake_A_BRandom,self.fake_B_ARandom)

    def update_D(self,image_a,image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward()

        #update discA
        self.discA_optimizer.zero_grad()
        loss_D1_A = self.backward_D(self.discA, self.real_A, self.fake_A_BA)
        self.disA_loss = loss_D1_A.item()
        self.discA_optimizer.step()

        # update disA2
        self.discA2_optimizer.zero_grad()
        loss_D2_A = self.backward_D(self.discA2, self.real_A_random, self.fake_A_BRandom)
        self.disA2_loss = loss_D2_A.item()
        if not self.no_ms:
            loss_D2_A2 = self.backward_D(self.discA2, self.real_A_random, self.fake_A_BRandom2)
            self.disA2_loss += loss_D2_A2.item()
        self.discA2_optimizer.step()

        # update disB
        self.discB_optimizer.zero_grad()
        loss_D1_B = self.backward_D(self.discB, self.real_B, self.fake_B_AB)
        self.disB_loss = loss_D1_B.item()
        self.discB_optimizer.step()

        # update disB2
        self.discB2_optimizer.zero_grad()
        loss_D2_B = self.backward_D(self.discB2, self.real_B_random, self.fake_B_ARandom)
        self.disB2_loss = loss_D2_B.item()
        if not self.no_ms:
            loss_D2_B2 = self.backward_D(self.discB2, self.real_B_random, self.fake_B_ARandom2)
            self.disB2_loss += loss_D2_B2.item()
        self.discB2_optimizer.step()

        # update disContent
        self.discContent_optimizer.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.discContent.parameters(), 5)
        self.discContent_optimizer.step()
        
    
    def backward_D(self,netD,real,fake):
        assert torch.any(torch.isnan(fake)) == False,print(fake)
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        lossD = 0
        for it,(out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake)
            all1 = torch.ones_like(out_real)
            ad_fake_loss = F.binary_cross_entropy(out_fake, all0)
            ad_true_loss = F.binary_cross_entropy(out_real, all1)
            lossD += ad_true_loss + ad_fake_loss
        lossD.backward()
        return lossD
    
    def backward_contentD(self, z_content_A, z_content_B):
        pred_fake = self.discContent.forward(z_content_A.detach())#(b,1) b=1
        pred_real = self.discContent.forward(z_content_B.detach())#(b,1)
        
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)#Size([1])
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones_like((out_real))
            all0 = torch.zeros_like((out_fake))
            ad_true_loss = F.binary_cross_entropy(out_real, all1)
            ad_fake_loss = F.binary_cross_entropy(out_fake, all0)
            loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D
    

    ####################### update encoder & generator #########################
    def update_EG(self):
        #update enc_c,enc_s,gen
        self.enc_c_optimizer.zero_grad()
        self.enc_s_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        self.backward_EG()
        self.backward_G_alone()
        self.enc_c_optimizer.step()
        self.enc_s_optimizer.step()
        self.gen_optimizer.step()

        
    
    def backward_EG(self):
        # content Ladv for generator : make content from two domain are similar
        loss_Acontent_adv = self.backward_content_adv(self.z_content_a)
        loss_Bcontent_adv = self.backward_content_adv(self.z_content_b)

        # Ladv for image generator
        loss_gan_adv_A = self.backward_GAN_adv(self.fake_A_BA,self.discA)
        loss_gan_adv_B = self.backward_GAN_adv(self.fake_B_AB,self.discB)

        # KL Loss
        if self.concat:
            kl_loss_zs_a = self.KL_Loss(self.mu_a,self.logvar_a)*0.01
            kl_loss_zs_b = self.KL_Loss(self.mu_b,self.logvar_b)*0.01
        else:
            kl_loss_zs_a = self._l2_regularize(self.z_style_a)*0.01
            kl_loss_zs_b = self._l2_regularize(self.z_style_b)*0.01
        
        kl_loss_zc_a = self._l2_regularize(self.z_content_a)*0.01
        kl_loss_zc_b = self._l2_regularize(self.z_content_b)*0.01

        # perception loss
        loss_G_L1_A = self.criterionL1(self.fake_A_BA,self.real_A)*5
        loss_G_L1_B = self.criterionL1(self.fake_B_AB,self.real_B)*5
        loss_G_L1_AA = self.criterionL1(self.fake_A_AA,self.real_A)*5
        loss_G_L1_BB = self.criterionL1(self.fake_B_BB,self.real_B)*5

        #SSIM
        ssim_BA = 1 - ssim(self.fake_A_BA, self.real_A,data_range=2.0)
        ssim_AB = 1 - ssim(self.fake_B_AB, self.real_B,data_range=2.0)

        ssim_loss = (ssim_AB + ssim_BA) * 10
       
        loss_G = loss_Acontent_adv + loss_Bcontent_adv + \
                 loss_gan_adv_A + loss_gan_adv_B + \
                 kl_loss_zs_a + kl_loss_zs_b + \
                 kl_loss_zc_a + kl_loss_zc_b + \
                 loss_G_L1_A + loss_G_L1_B + \
                 loss_G_L1_AA + loss_G_L1_BB + ssim_loss

        loss_G.backward(retain_graph = True)

        self.gan_loss_a = loss_gan_adv_A.item()
        self.gan_loss_b = loss_gan_adv_B.item()
        self.gan_loss_acontent = loss_Acontent_adv.item()
        self.gan_loss_bcontent = loss_Bcontent_adv.item()
        self.kl_loss_zs_a = kl_loss_zs_a.item()
        self.kl_loss_zs_b = kl_loss_zs_b.item()
        self.kl_loss_zc_a = kl_loss_zc_a.item()
        self.kl_loss_zc_b = kl_loss_zc_b.item()
        self.l1_recon_A_loss = loss_G_L1_A.item()
        self.l1_recon_B_loss = loss_G_L1_B.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.ssimloss = ssim_loss.item()
        self.G_loss = loss_G.item()

    def backward_G_alone(self):
        # Ladv for generator
        loss_G_GAN2_A = self.backward_GAN_adv(self.fake_A_BRandom, self.discA2)
        loss_G_GAN2_B = self.backward_GAN_adv(self.fake_B_ARandom, self.discB2)
        if not self.no_ms:
            loss_G_GAN2_A2 = self.backward_GAN_adv(self.fake_A_BRandom2, self.discA2)
            loss_G_GAN2_B2 = self.backward_GAN_adv(self.fake_B_ARandom2, self.discB2)

        # mode seeking loss for A-->B and B-->A
        if not self.no_ms:
            lz_AB = torch.mean(torch.abs(self.fake_B_ARandom2 - self.fake_B_ARandom)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
            lz_BA = torch.mean(torch.abs(self.fake_A_BRandom2 - self.fake_A_BRandom)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
            eps = 1e-5
            loss_lz_AB = 1 / (lz_AB + eps)
            loss_lz_BA = 1 / (lz_BA + eps)
        # latent regression loss
        if self.concat:
            loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
        else:
            loss_z_L1_a = torch.mean(torch.abs(self.z_s_random_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self.z_s_random_b - self.z_random)) * 10

        # loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
        if not self.no_ms:
            loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
            loss_z_L1 += (loss_lz_AB + loss_lz_BA)
        loss_z_L1.backward()
        self.l1_recon_z_loss_a = loss_z_L1_a.item()
        self.l1_recon_z_loss_b = loss_z_L1_b.item()
        if not self.no_ms:
            self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
            self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
            self.lz_AB = loss_lz_AB.item()
            self.lz_BA = loss_lz_BA.item()
        else:
            self.gan2_loss_a = loss_G_GAN2_A.item()
            self.gan2_loss_b = loss_G_GAN2_B.item()

    def KL_Loss(self,mu,logvar):
        kl = -0.5 * torch.sum(logvar + 1 - mu.pow(2) - logvar.exp())
        return kl
    
    def _l2_regularize(self,mu):
        mu_2 = torch.pow(mu,2)
        return torch.mean(mu_2)

    # make content disc can not distinguish the content from domain A or from domain B
    def backward_content_adv(self,z_content):
        outs = self.discContent.forward(z_content)
        for logit in outs:
            fake = torch.sigmoid(logit)
            all_half = 0.5*torch.ones((fake.size(0))).cuda(self.gpu)
            adv_loss = F.binary_cross_entropy(fake,all_half)
        return adv_loss
    
    def backward_GAN_adv(self,fake,disc):
        out_fake = disc.forward(fake)
        loss_G = 0
        for out in out_fake:
            output_fake = torch.sigmoid(out)
            all_ones = torch.ones_like(output_fake)
            loss_G += F.binary_cross_entropy(output_fake,all_ones)
        return loss_G

    def save_model(self,filename,ep,total_it):
        state = {
            'enc_c':self.enc_c.state_dict(),
            'enc_s':self.enc_s.state_dict(),
            'gen':self.gen.state_dict(),
            'discA': self.discA.state_dict(),
            'discA2': self.discA2.state_dict(),
            'discB': self.discB.state_dict(),
            'discB2': self.discB2.state_dict(),
            'discContent': self.discContent.state_dict(),
            'MI_estimator': self.MI_estimation.state_dict(),
            'enc_c_optimizer': self.enc_c_optimizer.state_dict(),
            'enc_s_optimizer': self.enc_s_optimizer.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'discA_optimizer': self.discA_optimizer.state_dict(),
            'discB_optimizer': self.discB_optimizer.state_dict(),
            'discA2_optimizer': self.discA2_optimizer.state_dict(),
            'discB2_optimizer': self.discB2_optimizer.state_dict(),
            'discContent_optimizer': self.discContent_optimizer.state_dict(),
            'MI_estmator_optimizer': self.MI_estimation_optimizer.state_dict(),
            'epoch': ep,
            'total_it': total_it
        }
        torch.save(state,filename)
        return

    def resume(self,pth_file,train = True):
        import os
        if os.path.isfile(pth_file):
            print("=> loading checkpoint {}".format(pth_file))
            device = torch.device('cuda:{}'.format(self.gpu))
            checkpoint = torch.load(pth_file,map_location=device)
            if train:
                self.discA.load_state_dict(checkpoint['discA'])
                self.discA2.load_state_dict(checkpoint['discA2'])
                self.discB.load_state_dict(checkpoint['discB'])
                self.discB2.load_state_dict(checkpoint['discB2'])
                self.discContent.load_state_dict(checkpoint['discContent'])
                self.MI_estimation.load_state_dict(checkpoint['MI_estimator'])
            self.enc_c.load_state_dict(checkpoint['enc_c'])
            self.enc_s.load_state_dict(checkpoint['enc_s'])
            self.gen.load_state_dict(checkpoint['gen'])

            if train:
                self.discA_optimizer.load_state_dict(checkpoint['discA_optimizer'])
                self.discA2_optimizer.load_state_dict(checkpoint['discA2_optimizer'])
                self.discB_optimizer.load_state_dict(checkpoint['discB_optimizer'])
                self.discB2_optimizer.load_state_dict(checkpoint['discB2_optimizer'])
                self.discContent_optimizer.load_state_dict(checkpoint['discContent_optimizer'])
                self.enc_c_optimizer.load_state_dict(checkpoint['enc_c_optimizer'])
                self.enc_s_optimizer.load_state_dict(checkpoint['enc_s_optimizer'])
                self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
                self.MI_estimation_optimizer.load_state_dict(checkpoint['MI_estmator_optimizer'])
            print('Completed! Resuming from epoch {},total_iter:{}.'.format(checkpoint['epoch'],checkpoint['total_it']))
            return checkpoint['epoch'],checkpoint['total_it']
        else:
            print('=> no checkpoint found at {}' .format(pth_file))
            