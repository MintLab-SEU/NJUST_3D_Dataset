from easydict import EasyDict

C = EasyDict()
# Random Seed
C.seed = 10288
C.deterministic = False

# Load & Dump Path
C.data_dir = 'dataset/raw_skeleton.pkl'
C.ckpt_dir = 'checkpoints'
C.expt_dir = 'experiments'

C.model = EasyDict()
# Model
C.model.spatial_dim = 75
C.model.input_len = 30
C.model.num_spatial_blocks = 1
C.model.num_temporal_blocks = 20
C.model.layer_norm_axis = 'spatial'  # in ['all', 'spatial', 'temporal', 'relu', None]

C.train = EasyDict()
# Train in main()
C.train.type = 'epoch'  # in ['epoch', 'iter']
C.train.max_epoch = 80
C.train.max_iter = 5000000
C.train.warm_up = True
C.train.save_state_intervals = 10
C.train.save_last_state = True
C.train.save_best_state = True

# DataLoader
C.train.batch_size = 256
C.train.data_aug = False
C.train.input_len = C.model.input_len
C.train.output_len = 6

# Loss Function
C.train.use_dct = True
C.train.cut_off = 0.8
C.train.use_relative = True
C.train.p_factor = 1.0
C.train.use_v_loss = True
C.train.v_factor = 1.0
C.train.use_a_loss = True
C.train.a_factor = 1.0

C.eval = EasyDict()
# Eval
C.eval.input_len = C.model.input_len
C.eval.output_len = C.train.output_len
C.eval.test_len = 15
C.eval.test_all = False
C.eval.num_per_action = 256
C.eval.batch_size = C.eval.num_per_action
C.eval.seed = 1234

C.optim = EasyDict()
# Optimizer
C.optim.type = 'Adam'
C.optim.lr = 0.0003
C.optim.weight_decay = 0.0001

# Learning Rate Decay Scheduler
C.optim.decay_mode = 'Linear'
C.optim.lr_min = 0.00001
C.optim.decay_total_steps = 70
C.optim.use_refine = True

config = C
cfg = C

if __name__ == '__main__':
    print(config)
