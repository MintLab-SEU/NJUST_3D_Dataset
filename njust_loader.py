import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class NJUST3D_Dataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 data_aug: bool = False,
                 observe_len: int = 30,
                 predict_len: int = 6,
                 shift_step: int = 1,
                 eval_all_frames: bool = False,
                 eval_num_per_action: int = 256,
                 eval_seed: int = 10288):
        super(NJUST3D_Dataset, self).__init__()
        self.data_dir = data_dir

        assert split in ['train', 'val', 'test', 'this_val']
        self.split = split
        self.data_aug = False if split == 'test' else data_aug

        self.used_joint_indexes = np.array([i for i in range(25)], dtype=int)

        self.njust3d_files = self._get_files()

        self.njust3d_motion_input_length = observe_len
        self.njust3d_motion_target_length = predict_len

        self.motion_dim = len(self.used_joint_indexes) * 3
        self.shift_step = shift_step

        self.njust3d_files_samples = []
        if split == 'test':
            self._find_eval_samples(eval_num_per_action, eval_seed)
        else:
            self._collect_all()

        self.file_length = len(self.data_idx)

    def __len__(self):
        return self.file_length

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame,
                                  start_frame + self.njust3d_motion_input_length + self.njust3d_motion_target_length,
                                  1)
        motion = self.njust3d_seqs[idx][frame_indexes]

        if self.data_aug:
            if torch.rand(1)[0] > 0.5:
                idx = [i for i in range(motion.size(0) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]

        njust3d_motion_input = motion[:self.njust3d_motion_input_length]  # meter
        njust3d_motion_target = motion[self.njust3d_motion_input_length:]  # meter

        return njust3d_motion_input, njust3d_motion_target

    def _get_files(self):

        seq_names = []
        if self.split == 'train':
            seq_names = ["R002", "R003"]
        elif self.split == 'this_val':
            seq_names = ["R001"]
        elif self.split == 'val':
            seq_names = ["S11"]
        elif self.split == 'test':
            seq_names = ["R001"]
        else:
            raise NotImplementedError

        with open(self.data_dir, 'rb') as f:
            file_list = pickle.load(f)

        njust3d_files = []
        for data in file_list:
            if data['name'][12:16] not in seq_names:
                continue
            num_frames = data['num_frames']
            xyz_info = data['data']['BodyID']['joint'].reshape(-1, 25, 3)
            point = xyz_info[0, 1].copy()
            xyz_info -= point
            # if not self.split == 'test':
            #     xyz_info = xyz_info[:, self.used_joint_indexes, :]
            njust3d_files.append(xyz_info.reshape(-1, 75))
        return njust3d_files

    def _collect_all(self):

        self.njust3d_seqs = []
        self.data_idx = []
        idx = 0

        for njust3d_motion_poses in self.njust3d_files:
            num_frames = len(njust3d_motion_poses)
            if num_frames < self.njust3d_motion_target_length + self.njust3d_motion_input_length:
                continue

            self.njust3d_seqs.append(njust3d_motion_poses)

            # Collect all samples
            valid_start_frames = np.arange(0,
                                           num_frames - self.njust3d_motion_input_length - self.njust3d_motion_target_length + 1,
                                           self.shift_step)

            self.njust3d_files_samples.append(len(valid_start_frames))

            self.data_idx.extend(zip([idx] * len(valid_start_frames), valid_start_frames.tolist()))
            idx += 1

    def _find_eval_samples(self, num_per_action=256, np_seed=10288):

        self.njust3d_seqs = self.njust3d_files
        self.data_idx = []
        idx = 0

        index = np.arange(0, len(self.njust3d_files), 1).reshape(-1, 18)

        for action_id in range(18):
            action_data_idx = []
            for file_index in index[:, action_id]:
                num_frames = len(self.njust3d_files[file_index])
                if num_frames < self.njust3d_motion_target_length + self.njust3d_motion_input_length:
                    continue

                # Collect all samples
                valid_start_frames = np.arange(0,
                                               num_frames - self.njust3d_motion_input_length - self.njust3d_motion_target_length + 1,
                                               self.shift_step)

                action_data_idx.extend(zip([file_index] * len(valid_start_frames), valid_start_frames.tolist()))

            # Random sampling of several samples
            rs = np.random.RandomState(np_seed)
            rs.shuffle(action_data_idx)
            self.data_idx.extend(action_data_idx[:num_per_action])


