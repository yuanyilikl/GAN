import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.autograd.grad_mode import F
import torch.nn.functional as nnF
import torchvision 
from torchvision import transforms  
from torch.utils import data
from torchvision.transforms.transforms import ToTensor
import tqdm
import os

class Config(object):

    num_workers  = 4
    image_size   = 28
    h_dim        = 400
    z_dim        = 20
    num_epochs   = 30
    lr           = 0.001
    batch_size   = 128

    use_gpu      = False
    gen_epoch    = None
    max_epoch    = 200

    train_every    = 1      
    save_every   = 5
    gen_search_num = 512

opt = Config()
device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

# model
class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(opt.image_size*opt.image_size, opt.h_dim)
        self.fc2 = nn.Linear(opt.h_dim, opt.z_dim)
        self.fc3 = nn.Linear(opt.h_dim, opt.z_dim)
        self.fc4 = nn.Linear(opt.z_dim, opt.h_dim)  # 20*400
        self.fc5 = nn.Linear(opt.h_dim, opt.image_size*opt.image_size)#400*784

    def encode(self, x):
        h = nnF.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = nnF.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h)) # 128*784
    
    def forward(self, x):
        x = x.view(x.size(0), 784)
        mu, log_var = self.encode(x) # 400*20
        z = self.reparameterize(mu, log_var) # 400*20
        x_reconst = self.decode(z)  # 128*784
        return x_reconst, mu, log_var

def makedir(opt):
    try:
        os.mkdir("./figure")
    except:
        pass
    try:
        os.mkdir("./checkpoints")
    except:
        pass

def loss_function(x_reconst, x, mu, log_var):
    reconst_loss = nnF.binary_cross_entropy(x_reconst, x.view(-1, opt.image_size*opt.image_size), size_average = False)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = reconst_loss + kl_div
    return loss

# main
def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    makedir(opt)   

    # work with data 
    dataset = torchvision.datasets.MNIST(root = "data", 
                                        train = True, 
                                        transform = transforms.ToTensor(), 
                                        download = False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size = opt.batch_size,
                                            shuffle = True, 
                                            num_workers = opt.num_workers, 
                                            drop_last = True)
    
    # net
    map_location = lambda storage, loc: storage
    model = VAE(opt).to(device) 
    if opt.gen_epoch:
        vae_path  = 'checkpoints/vae_net_%s.pth' %opt.gen_epoch
        model.load_state_dict(torch.load(vae_path, map_location = map_location))
    
    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)


    # train net
    epochs = range(opt.max_epoch)
    loss = 0
    for epoch in iter(epochs):
        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            x = Variable(x)
            x.to(device)

            if (ii + 1) % opt.train_every == 0:
                optimizer.zero_grad()
                x_reconst, mu, log_var = model(x)
                loss = loss_function(x_reconst, x, mu, log_var)
                loss.backward()
                optimizer.step()
        print("loss", loss)

        if (epoch+1) % opt.save_every == 0:
            #  model
            if opt.gen_epoch:
                cur_epoch = epoch + opt.gen_epoch
            else:
                cur_epoch = epoch
            vae_path = 'checkpoints/vae_net_%s.pth' %cur_epoch
            torch.save(model.state_dict(), vae_path)

# generater
@torch.no_grad()
def generate(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    dataset = torchvision.datasets.MNIST(root = "data", 
                                        train = True, 
                                        transform = transforms.ToTensor(), 
                                        download = False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size = opt.batch_size,
                                            shuffle = True, 
                                            num_workers = opt.num_workers, 
                                            drop_last = True)

    model = VAE(opt).to(device) 
    noises = torch.randn(opt.batch_size, opt.z_dim)
    noises = Variable(noises).to(device)
    if opt.gen_epoch is None:
        gen_epoch = 1
    else:
        gen_epoch = opt.gen_epoch
    ave_path  = 'checkpoints/vae_net_%s.pth' %gen_epoch

    map_location = lambda storage, loc: storage
    model.load_state_dict(torch.load(ave_path, map_location = map_location))
    model.to(device)

    #  generate imags 
    x, _ = iter(dataloader).next()
    x = Variable(x)
    x.to(device)
    out, _, _ = model(x)
    x_reconst = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)

    out2 = model.decode(noises).view(-1, 1, 28, 28)
    # save image
    gen_img = "result_vae_%ss.png" %opt.gen_epoch
    torchvision.utils.save_image((x_reconst), gen_img)
    
    gen_img = "sample_vae_%ss.png" %opt.gen_epoch
    torchvision.utils.save_image((out2), gen_img)


if __name__ == '__main__':
    import fire
    fire.Fire()

  







