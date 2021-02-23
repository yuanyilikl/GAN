import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import visdom
import torchvision 
from torchvision import transforms  
from torch.utils import data
import tqdm
import os

class Config(object):
    #data_path    = "data/"
    num_workers  = 4
    image_size   = 28
    batch_size   = 128
    max_epoch    = 200
    lr_d         = 2e-4   # Discriminator learning rate
    lr_g         = 2e-4   # generator  learning rate
    beta1        = 0.5    # Adam optimizer's parameter--beta1
    use_gpu      = False
    nz           = 100    #  nosie dimension
    embe_dim     = 10

    save_path    = "figure/"
    vis          = False  # use visdom or not
    env          = "GAN"  # visdom's env
    plot_every   = 20 
    save_every   = 2 
    
    debug_file   = "/tem/debuggan"  # if have this file and star debug
    dis_every    = 1      # every "dis_every" batch train discriminator
    gen_every    = 1      # every "gen_every" batch train generator
    decay_every  = 20

    gen_num      = 8*8
    gen_search_num = 512  # save "gen_num" picture in "gen_search_num"
    gen_mean     = 0
    gen_std      = 1
    gen_epoch    = None
    digit        = 8
    ncateg       = 10

opt = Config()
device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

# model
class NetG(nn.Module):
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.label_emb = nn.Embedding(10, opt.embe_dim)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)   

class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        self.label_emb = nn.Embedding(10, opt.embe_dim) 

        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
          
def makedir(opt):
    try:
        os.mkdir("./figure")
    except:
        pass
    try:
        os.mkdir("./checkpoints")
    except:
        pass

#  main
def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    makedir(opt)  

    # work with data 
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Scale(opt.image_size),
        torchvision.transforms.CenterCrop(opt.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5),(0.5))
    ])
    dataset = torchvision.datasets.MNIST(root = "data", train = True, transform = trans, download = False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size = opt.batch_size,
                                            shuffle = True, 
                                            num_workers = opt.num_workers, 
                                            drop_last = True)

    #  net
    map_location = lambda storage, loc: storage
    netg, netd = NetG(opt), NetD(opt)
    if opt.gen_epoch:
        netg_path    = 'checkpoints/cgan_netg_%s.pth' %opt.gen_epoch
        netd_path    = 'checkpoints/cgan_netd_%s.pth' %opt.gen_epoch
        netd.load_state_dict(torch.load(netd_path, map_location = map_location))
        netg.load_state_dict(torch.load(netg_path, map_location = map_location))
        # python mnist_dcgan.py train --gen_epoch=3
    netd.to(device)
    netg.to(device)

    # optimizer and loss
    optimizer_g = torch.optim.Adam(netg.parameters(), opt.lr_g, betas = (opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(netd.parameters(), opt.lr_d, betas = (opt.beta1, 0.999))
    criterion = torch.nn.BCELoss().to(device)

    # true: label = 1, fake: label = 0
    true_labels = Variable(torch.ones(opt.batch_size)).to(device)
    fake_labels = Variable(torch.zeros(opt.batch_size)).to(device)
    fix_noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1)).to(device)
    noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1)).to(device)

    # train net
    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        for ii, (img, label ) in tqdm.tqdm(enumerate(dataloader)):
            real_img = Variable(img)
            real_label = Variable(label)
            real_img.to(device) 
            real_label.to(device)                                                         
            
            # train discriminator
            if (ii + 1) % opt.dis_every == 0:
                optimizer_d.zero_grad()

                output = netd(real_img, real_label)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_label = torch.randint(0, 10, (opt.batch_size, )).to(device)
                fake_img = netg(noises, fake_label).detach()
                fake_output = netd(fake_img, fake_label)
                error_d_fake = criterion(fake_output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

            # train generator
            if (ii + 1) % opt.gen_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_label = torch.randint(0, 10, (opt.batch_size, )).to(device)
                fake_img = netg(noises, fake_label)               
                fake_output = netd(fake_img, fake_label)
                error_g = criterion(fake_output, true_labels)
                error_g.backward()
                optimizer_g.step() 

        if (epoch+1) % opt.save_every == 0:
            # save image and model
            if opt.gen_epoch:
                cur_epoch = epoch + opt.gen_epoch
            else:
                cur_epoch = epoch
            netg_path    = 'checkpoints/cgan_netg_%s.pth' %cur_epoch
            netd_path    = 'checkpoints/cgan_netd_%s.pth' %cur_epoch
            torch.save(netd.state_dict(), netd_path)
            torch.save(netg.state_dict(), netg_path)



# generater
@torch.no_grad()
def generate(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    noises = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = Variable(noises, volatile = True).to(device)
    label  = opt.digit * torch.ones(opt.gen_search_num).long().to(device)
    if opt.gen_epoch is None:
        gen_epoch = 1
    else:
        gen_epoch = opt.gen_epoch
    netg_path    = 'checkpoints/cgan_netg_%s.pth' %gen_epoch
    netd_path    = 'checkpoints/cgan_netd_%s.pth' %gen_epoch



    map_location = lambda storage, loc: storage
    netd.load_state_dict(torch.load(netd_path, map_location = map_location))
    netg.load_state_dict(torch.load(netg_path, map_location = map_location))
    netd.to(device)
    netg.to(device)

    #  generate imags and compute scores
    fake_img = netg(noises, label)
    scores = netd(fake_img, label).detach()
 
    # choice best image
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii].view(1,opt.image_size, opt.image_size))

    # save image
    gen_img = "figure/result_cgan_%s.png" %opt.digit
    torchvision.utils.save_image(torch.stack(result), 
                                            gen_img, 
                                            normalize = True, 
                                            range=(-1, 1))

if __name__ == '__main__':
    import fire
    fire.Fire()



  







