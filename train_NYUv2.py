import time
import torch
import torch.nn as nn
import cv2
import Mymodel
from nyu_dataloader import *
from options import opts
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os
import numpy as np


def init_opts(opts):
    opts.device = torch.device("cuda")
    opts.results_dir = "./results/NYU8"
    if not opts.is_debug:
        os.makedirs(opts.results_dir, exist_ok=True)
    return opts

np.random.seed(opts.seed)
torch.manual_seed(opts.seed)
torch.cuda.manual_seed_all(opts.seed)

def calc_rmse(aa, bb, minmax):
    aa = aa[6:-6, 6:-6]
    bb = bb[6:-6, 6:-6]
    aa = aa * (minmax[0] - minmax[1]) + minmax[1]
    bb = bb * (minmax[0] - minmax[1]) + minmax[1]
    return np.sqrt(np.mean(np.power(aa - bb, 2)))

@torch.no_grad()
def validate(net, root_dir='/test/yjy/myGAN/Middlebury3.0/NYU8-my/nyu_data'):
    data_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = NYU_v2_dataset(root_dir=root_dir, transform=data_transform, train=False)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    net.eval()
    rmse = np.zeros(449)
    test_minmax = np.load('%s/test_minmax.npy' % root_dir)
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        minmax = test_minmax[:, idx]
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        out, _ = net(target, guidance)
        rmse[idx] = calc_rmse(gt[0, 0].cpu().numpy(), out[0, 0].cpu().numpy(), minmax)
        # if idx % 20 == 0:
        #     cv2.imwrite('output_test%d.png' % idx, out[0, 0].cpu().detach().numpy() * 255)
        #     cv2.imwrite('output_target%d.png' % idx, target[0, 0].cpu().detach().numpy() * 255)
        #     cv2.imwrite('output_gt%d.png' % idx, gt[0, 0].cpu().detach().numpy() * 255)
        t.set_description('[validate] rmse: %f' % rmse[:idx + 1].mean())
        t.refresh()
    return rmse.mean()

def main(opts):
    opts = init_opts(opts)
    net = Mymodel.Model(opts).cuda()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=opts.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=80000, gamma=0.5)
    data_transform = transforms.Compose([transforms.ToTensor()])
    nyu_dataset = NYU_v2_dataset(root_dir='/test/yjy/myGAN/Middlebury3.0/NYU8-my/nyu_data', transform=data_transform)
    dataloader = torch.utils.data.DataLoader(nyu_dataset, batch_size=20, shuffle=True)
    rmse_best, epoch_best = 10000, -1

    validate(net)
    for epoch in range(opts.epoch):
        net.train()
        running_loss = 0.0
        loader_tqdm = tqdm(iter(dataloader), leave=True, total=len(dataloader))
        for idx, data in enumerate(loader_tqdm):
            scheduler.step()
            guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
            out1, out = net(target, guidance)
            loss = criterion(out, gt) + criterion(out1, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            if idx % 50 == 0:
                loader_tqdm.set_description('[train epoch:%d] loss: %.8f' % (epoch + 1, running_loss / 50.0))
                loader_tqdm.refresh()

        if epoch % 5 == 0 and epoch != 0:
            rmse = validate(net)
            if rmse < rmse_best:
                rmse_best = rmse
                epoch_best = epoch
                torch.save(net.state_dict(), os.path.join(opts.output_model, "NYUmodelbest%f_8yjy%d.pth" % (rmse_best, epoch_best)))
            print('[best epoch :%d] rmse:%.5f   [current epoch :%d] rmse:%.5f'  % (epoch_best, rmse_best, epoch, rmse))

if __name__ == '__main__':
    main(opts)
