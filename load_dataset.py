import argparse
from njust_loader import NJUST3D_Dataset
from dataset.default import cfg
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Command-line argument are used for future skeleton generation network')

parser.add_argument('-i', '--index', type=int, default=0,help='index number of the experiment')
parser.add_argument('-d', '--description', type=str, default='no description.',help='additional description of experimental details')
parser.add_argument('--only_eval', action='store_true', default=True,help='only evaluate model performance, instead of training')
parser.add_argument('--state_load_path', type=str,help='path to loading the model state dictionary')
args = parser.parse_args()

class Process:
    def __init__(self):
        self._create_loader()

    def _create_loader(self):
        if not args.only_eval:
            self.train_loader = DataLoader(
                NJUST3D_Dataset(cfg.data_dir,
                            split='train',
                            data_aug=cfg.train.data_aug,
                            observe_len=cfg.train.input_len,
                            predict_len=cfg.train.output_len),
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True
            )
            self.val_loader = DataLoader(
                NJUST3D_Dataset(cfg.data_dir,
                            split='this_val',
                            data_aug=False,
                            observe_len=cfg.train.input_len,
                            predict_len=cfg.train.output_len),
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
        else:
            self.train_loader = None
            self.val_loader = None

        test_dataset = NJUST3D_Dataset(cfg.data_dir,
                                   split='test',
                                   data_aug=False,
                                   observe_len=cfg.eval.input_len,
                                   predict_len=cfg.eval.test_len,
                                   eval_all_frames=cfg.eval.test_all,
                                   eval_num_per_action=cfg.eval.num_per_action,
                                   eval_seed=cfg.eval.seed)

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        print('OK')

if __name__ == "__main__":
    Process()