import torch
import torch.nn as nn
import Mymodel.RSAG as RSAG

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        print('Making model...')
        self.opt = opt
        self.scale = opt.scale
        self.device = opt.device
        self.model = RSAG.make_model(opt).to(self.device)
        self.n_GPUs = 1
        self.load('/test/yjy/myGAN/2021CVPR/NYU8/results/NYUmodelbest0.025074_8yjy140.pth', pre_train=opt.pre_train)

    def forward(self, x,rgb):
        return self.model(x,rgb)

    def load(self, path, pre_train=False):
        kwargs = {}
        if pre_train != False:
            self.get_model().load_state_dict(torch.load(path, **kwargs),strict=False)
            print('Loading model from {}'.format(pre_train))

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
