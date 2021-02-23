import os
import models
from data.dataset import DogCat
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import torch
import visdom
import time
from torchvision import transforms
from tqdm import tqdm
from torchnet import meter

class DefaultCongig(object):
    env              = "default"               # visdom 
    model            = "AlexNet"               # model type
    #model            = "ResNet" 
    train_data_root  = "./data/train/"         # path of training set
    test_data_root   = "./data/test"          # path of testing set
    #load_model_path  = "checkpoints/model.pth" # path of loading model, (None = not loading)
    load_model_path  = None # path of loading model, (None = not loading)

    batch_size       = 128                     # batch size
    use_gpu          = False                    # use gpu or not
    num_workers      = 4                       # how many workers for loading data
    print_freq       = 20                      # print info every N batch

    debug_file       = "/tem/debug"            # if os.path.exists(debug file):enter ipdb
    result_file      = "result.csv"          

    max_epoch        = 10                      
    lr               = 0.1                     # learning rate 
    lr_decay         = 0.95                    # when val_loss increase, lr = lr*lr_decay
    weight_decay     = 1e-4                    # loss function


opt = DefaultCongig()

class DogCat(data.Dataset):
    def __init__(self, root, transforms0 = None, train = True, test = False):
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.test:
            imgs = sorted(imgs, key = lambda x: x.split('.')[-2].split('/')[-1])
        else:
            imgs = sorted(imgs, key = lambda x: x.split('.')[-2])
    
        imgs_num = len(imgs)    
        # test set , train set:vali set = 7:3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[: int(0.7 * imgs_num)]
        else :
            self.imgs = imgs[int(0.7 * imgs_num):]    
        print(len(imgs))
        #  normalize
        if transforms0 is None:
            normalize = transforms.Normalize([0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            #  test set, validation set
            if self.test or not train:
                self.transforms0 = transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), 
                                                      transforms.ToTensor(), normalize])
            #  train set
            else:
                self.transforms0 = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224),
                                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),normalize])
       
    def __gititem__(self, index):
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split(".")[-2].split("/")[-1])
        else:
            label = 1 if "dog" in imgs_path.split("/")[-1] else 0
        
        data = Image.open(img_path)
        data = self.transforms0(data)
        
        return data, label
        
    def __len__(self):
        return len(self.imgs)


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def save(self, name = None):
        if name is None:
            prefix = "checkpoints/" + self.module_name + "_"
            name = time.strftime(prefix + "%m%d_%H: %M: %S.pth")
        torch.save(self.state_dict(), name)
        return name


class Visualizer(object):
    def __init__ (self, env = "default", **kwargs):
        # save("loss", 23)
        self.vis = visdom.Visdom(env = env, **kwargs)
        self.index = {}
        self.log_text = ""
    
    def reinit (self, env = "default", **kwargs):
        # modifly visdom
        self.vis = visdom.Visdom(env = env, **kwargs)
        return self
    
    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)
            
    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v) 
    
    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y = np.array([y]), x = np.array([x]), win = (name), opts = dict(title = name), 
                      updata = None if x == o else "append", **kwargs)
        self.index[name] = x + 1
        
    def img(self, name, img_, **kwargs):
        self.vis.images(img_.cpu().numpy(), win = (name), opts = dict(title = name), **kwargs)
        
    def log(self, info, win = "log_text"):
        self.log_text += ("[{time}]{info}<br>".format(time.time.strftime("%m%d_%H%M%S"), info = info))
        self.vis.text(self.log_text, win)

    def __getatter__ (self, name):
        return getatter(self.vis, name)


# mian
def train(**kwargs):
    #opt.parse(kwargs)
    vis = Visualizer(opt.env)
    
    # step 1 : model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
        
    # step 2 : data
    train_data = DogCat(opt.train_data_root, train = True)
    val_data = DogCat(opt.train_data_root, train = False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle = True, num_workers = opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle = False, num_workers = opt.num_workers)
    
    #  step 3 : Objective function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = opt.weight_decay)
    
    #  step 4 : statistical index 
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10
    
    # step 5 : trian
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        
        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            
            # train parameters
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            optimizer.step()
            
            # update parameters and visualization
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)
            
            if ii%opt.print_freq == opt.print_freq-1:
                vis.plot("loss", loss_meter.value()[0])
                
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace
        model.save()
        
        # index in valitation set  and visualization
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot("val_accuracy", val_accuracy)
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}"
                .format(epoch = epoch, loss = loss_meter.value()[0], val_cm = str(val_cm.value()),
                        train_cm = str(confusion_matrix.value()), lr = lr))
        
        # If loss function is not decreasing, reducing lr
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                
        previous_loss = loss_meter.value()[0]
    

def val(model, dataloader):
    
    model.eval()
    
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile = True)
        val_label = Variable(label.long(), volatile = True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeezer(), label.long())
        
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1])/(cm_value.sum())
    
    return confusion_matrix, accuracy
    

def test(**kwargs):
    opt.parse(kwargs)
    
    # model
    model = getatter(models, opt.models.eval())
    if opt.load_model_path:
        model.loca(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    
    # data
    train_data = DogCat(opt.test_data_root, test = True)
    test_dataloader = DataLoader(train_data, batch_size = opt.batch_size, 
                                 shuffle = False, num_workers = opt.num_workers)
    
    results = []    
    for ii, (data, label) in enumerate(test_dataloader):
        input = torch.autograd.Variable(data, volatile = True)
        if opt.use_gpu: 
            input = input.cuda
        score = model(input)
        probability = torch.nn.functional.softmax(score)[:, 1].data.tolist()
        batch_results = [(path_,probability_)
            for path_, probability_ in zip(path, probability)]
        results += batch_results
    
    write_csv(results, opt.result_file)
    return results


def help():
    print("""
    usage : python {0} <function> [--args = value, ]
    <function> := train | test | help
    example:
        python {0} train --env"env0701" -- lr = 0.01
        python {0} test -- dataset = "path/to/dataset/root/"
        python {0} help
    avaible args:""".format(__file__))
    
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()

