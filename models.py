import os
import torch
import torch.nn as nn
import itertools
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.to(device)
        # We extract features from relu2_2 (layer index 8 in torchvision vgg16 features)
        self.slice1 = torch.nn.Sequential()
        for x in range(9):
            self.slice1.add_module(str(x), vgg[x])
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        # VGG expects images in [0, 1] range and normalized by ImageNet stats
        # CycleGAN outputs are in [-1, 1]
        x_norm = (x + 1) / 2.0
        y_norm = (y + 1) / 2.0
        
        # Approximate ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_norm = (x_norm - mean) / std
        y_norm = (y_norm - mean) / std

        hx = self.slice1(x_norm)
        hy = self.slice1(y_norm)
        return self.criterion(hx, hy)


class ArcFaceLoss(nn.Module):
    def __init__(self, device):
        super(ArcFaceLoss, self).__init__()
        # Load pretrained InceptionResnetV1 on VGGFace2
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        for param in self.parameters():
            param.requires_grad = False
            
        # We can use Cosine Embedding Loss as distance metric
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, x, y):
        # CycleGAN images are in [-1, 1]. InceptionResnetV1 also expects [-1, 1] for best results, but
        # assumes images are cropped out faces roughly 160x160.
        # We resize features via interpolation before passing if they are too large or small,
        # but 256x256 is usually fine since it has adaptive pooling at the end.
        x_resized = torch.nn.functional.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        y_resized = torch.nn.functional.interpolate(y, size=(160, 160), mode='bilinear', align_corners=False)
        
        feat_x = self.model(x_resized)
        feat_y = self.model(y_resized)
        
        # Targets for CosineEmbeddingLoss: 1 indicates they should be similar
        target = torch.ones(feat_x.size(0)).to(x.device)
        return self.criterion(feat_x, feat_y, target)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResNetGenerator, self).__init__()

        # Initial Conv
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        # ResNet Blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=False),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]

        # Output Conv
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class CycleGANModel(nn.Module):
    def __init__(self, opt):
        super(CycleGANModel, self).__init__()
        self.opt = opt
        
        # GPU設定の解析とデバイスの決定
        gpu_ids = opt.gpu_ids.split(',') if hasattr(opt, 'gpu_ids') and opt.gpu_ids else []
        if len(gpu_ids) > 0 and gpu_ids[0] != '-1' and torch.cuda.is_available():
            gpu_id = int(gpu_ids[0])
            self.device = torch.device(f'cuda:{gpu_id}')
            print('========== GPU情報 ==========')
            print('[INFO] GPUを使用して学習・推論を行います')
            print(f'[INFO] デバイス: cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})')
            print('=============================')
        else:
            self.device = torch.device('cpu')
            print('========== GPU情報 ==========')
            print('[WARN] GPUが使用されていません。CPUで実行します。')
            if not torch.cuda.is_available():
                print('[WARN] (原因: torch.cuda.is_available() が False です。PyTorchがGPUを認識していません)')
            print('=============================')
            
        self.isTrain = opt.phase == 'train'

        print("=========================================")
        print(f"Device configuration: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
        else:
            print(f"WARNING: GPU is not used. CUDA Available: {torch.cuda.is_available()}, opt.gpu_ids: {opt.gpu_ids}")
        print("=========================================")

        # Generators
        self.netG_A = ResNetGenerator(opt.input_nc, opt.output_nc, opt.ngf, n_blocks=9 if opt.net_g == 'resnet_9blocks' else 6).to(self.device)
        self.netG_B = ResNetGenerator(opt.output_nc, opt.input_nc, opt.ngf, n_blocks=9 if opt.net_g == 'resnet_9blocks' else 6).to(self.device)

        # Discriminators & Losses
        if self.isTrain:
            self.netD_A = NLayerDiscriminator(opt.output_nc, opt.ndf, opt.n_layers_d).to(self.device)
            self.netD_B = NLayerDiscriminator(opt.input_nc, opt.ndf, opt.n_layers_d).to(self.device)

            # Standard Losses
            self.criterionGAN = nn.MSELoss()
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            
            # Extra Losses
            if self.opt.lambda_perceptual > 0.0:
                self.criterionPerceptual = PerceptualLoss(self.device)
            if self.opt.lambda_arcface > 0.0:
                self.criterionArcFace = ArcFaceLoss(self.device)

            # Optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = [self.optimizer_G, self.optimizer_D]
            
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)   # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)    # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)   # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)    # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), torch.ones_like(self.netD_A(self.fake_B)))
        
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), torch.ones_like(self.netD_B(self.fake_A)))
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Extra Losses primarily acting on Real -> Anime -> Reconstr Real (real_A and rec_A)
        self.loss_perceptual = 0
        if self.opt.lambda_perceptual > 0.0:
            self.loss_perceptual = self.criterionPerceptual(self.rec_A, self.real_A) * self.opt.lambda_perceptual

        self.loss_arcface = 0
        if self.opt.lambda_arcface > 0.0:
            self.loss_arcface = self.criterionArcFace(self.rec_A, self.real_A) * self.opt.lambda_arcface

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_perceptual + self.loss_arcface
        self.loss_G.backward()

    def compute_val_losses(self):
        """Computes all losses without backward() to be used during validation loop."""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        with torch.no_grad():
            # Identity loss
            if lambda_idt > 0:
                idt_A = self.netG_A(self.real_B)
                loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
                idt_B = self.netG_B(self.real_A)
                loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                loss_idt_A = torch.tensor(0.0).to(self.device)
                loss_idt_B = torch.tensor(0.0).to(self.device)

            # GAN loss D_A(G_A(A))
            loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), torch.ones_like(self.netD_A(self.fake_B)))
            
            # GAN loss D_B(G_B(B))
            loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), torch.ones_like(self.netD_B(self.fake_A)))
            
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
            
            # Extra Losses
            loss_perceptual = torch.tensor(0.0).to(self.device)
            if self.opt.lambda_perceptual > 0.0:
                loss_perceptual = self.criterionPerceptual(self.rec_A, self.real_A) * self.opt.lambda_perceptual

            loss_arcface = torch.tensor(0.0).to(self.device)
            if self.opt.lambda_arcface > 0.0:
                loss_arcface = self.criterionArcFace(self.rec_A, self.real_A) * self.opt.lambda_arcface

            # Discriminator Losses
            pred_real_A = self.netD_B(self.real_A)
            loss_D_real_A = self.criterionGAN(pred_real_A, torch.ones_like(pred_real_A))
            pred_fake_A = self.netD_B(self.fake_A.detach())
            loss_D_fake_A = self.criterionGAN(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_D_B = (loss_D_real_A + loss_D_fake_A) * 0.5
            
            pred_real_B = self.netD_A(self.real_B)
            loss_D_real_B = self.criterionGAN(pred_real_B, torch.ones_like(pred_real_B))
            pred_fake_B = self.netD_A(self.fake_B.detach())
            loss_D_fake_B = self.criterionGAN(pred_fake_B, torch.zeros_like(pred_fake_B))
            loss_D_A = (loss_D_real_B + loss_D_fake_B) * 0.5

            losses = {
                'G_A': loss_G_A.item(),
                'G_B': loss_G_B.item(),
                'Cyc_A': loss_cycle_A.item(),
                'Cyc_B': loss_cycle_B.item(),
                'D_A': loss_D_A.item(),
                'D_B': loss_D_B.item(),
            }
            if lambda_idt > 0.0:
                losses['idt_A'] = loss_idt_A.item()
                losses['idt_B'] = loss_idt_B.item()
            if self.opt.lambda_perceptual > 0.0:
                losses['VGG'] = loss_perceptual.item()
            if self.opt.lambda_arcface > 0.0:
                losses['Arc'] = loss_arcface.item()
                
            return losses

    def optimize_parameters(self):
        self.forward()      # compute fake images and reconstruction images.
        
        # G_A and G_B
        set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # D_A and D_B
        set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def save_networks(self, epoch):
        save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.netG_A.state_dict(), os.path.join(save_dir, f'{epoch}_net_G_A.pth'))
        torch.save(self.netG_B.state_dict(), os.path.join(save_dir, f'{epoch}_net_G_B.pth'))
        if self.isTrain:
            torch.save(self.netD_A.state_dict(), os.path.join(save_dir, f'{epoch}_net_D_A.pth'))
            torch.save(self.netD_B.state_dict(), os.path.join(save_dir, f'{epoch}_net_D_B.pth'))

    def load_networks(self, epoch):
        load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.netG_A.load_state_dict(torch.load(os.path.join(load_dir, f'{epoch}_net_G_A.pth'), map_location=self.device))
        self.netG_B.load_state_dict(torch.load(os.path.join(load_dir, f'{epoch}_net_G_B.pth'), map_location=self.device))
        if self.isTrain:
            try:
                self.netD_A.load_state_dict(torch.load(os.path.join(load_dir, f'{epoch}_net_D_A.pth'), map_location=self.device))
                self.netD_B.load_state_dict(torch.load(os.path.join(load_dir, f'{epoch}_net_D_B.pth'), map_location=self.device))
            except FileNotFoundError:
                print("Could not load discriminators. Skipping.")
