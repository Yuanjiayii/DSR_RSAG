from utils.attr_dict import AttrDict
import numpy as np

opts = AttrDict()
opts.seed = 123
opts.epoch = 1000
opts.output_model = 'results'
opts.pre_train = True

opts.n_blocks = 30
opts.n_colors = 1
opts.n_feats = 16
opts.lr = 0.00002
opts.factor = 8
opts.negval = 0.2

opts.min_rmse = 1000.0
opts.scale = [pow(2, s+1) for s in range(int(np.log2(opts.factor)))]
