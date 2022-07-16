from torch.utils.data import Dataset
import torch
import numpy as np
import os.path as osp
import random
import copy
import json
import pickle


class NTU_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, w_size, stride, dilate,
                 benchmark, data_split, label_num,
                 use_data_aug, view, data_part='all'):

        """
        Args:
            w_size: window size
            stride: sliding window stride
            label_num: 60/120
            data_split: train or test
            use_data_aug: flag for using data augmentation
        """
        assert data_split in ['train', 'val']
        self.data_fold = './data/NTU_{}'.format(label_num)

        self.w_size = w_size
        self.stride = stride
        self.dilate = dilate
        self.joint_num = 25
        self.use_data_aug = use_data_aug
        self.view = view

        self.joint_dim = 3

        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        print('loading NTU {} data......'.format(label_num))
        print('benchmark:', benchmark, 'data_split:', data_split, 'view:', self.view)
        print('use_data_aug: ',self.use_data_aug)


        seq_pth = osp.join(self.data_fold, '{}/{}_data.npy'.format(benchmark, data_split))
        self.seq_list = np.load(seq_pth, mmap_mode='r')

        print('{} seq num: {}'.format(data_split, len(self.seq_list)))


        info_pth = osp.join(self.data_fold, '{}/{}_video_info.json'.format(benchmark, data_split))
        with open(info_pth) as src_file:  # Unpickling
            self.video_info = json.load(src_file)

        #get label
        # label_pth = osp.join(self.data_fold, '{}/{}_label_norm.pkl'.format(benchmark, data_split))
        # with open(label_pth, 'rb') as f:
        #     self.label_list = pickle.load(f)[1]
        label_pth = osp.join(self.data_fold, '{}/{}_label.npy'.format(benchmark, data_split))
        self.label_list = np.load(label_pth)

        print(max(self.label_list), min(self.label_list))
        if label_num == 60:
            assert max(self.label_list) == 59 and min(self.label_list) == 0 and len(set(self.label_list)) == 60
        else:
            assert max(self.label_list) == 119 and min(self.label_list) == 0 and len(set(self.label_list)) == 120

        assert self.seq_list.shape[0] == len(self.video_info) == len(self.label_list)

        #load idx (semi-supversied learning)
        if data_split == 'train' and label_num == 60:
            idx_pth = './data/NTU_60_data_idx' + '/{}/{}_idx_list.pkl'.format(benchmark, data_part)
            with open(idx_pth, 'rb') as src_file:
                self.idx_list = pickle.load(src_file)
            if data_part == 'all':
                assert len(self.idx_list) == len(self.seq_list)
            if data_part == 'ten':
                len(self.idx_list) == round(len(self.seq_list) * 0.1)
        else:
            self.idx_list = list(range(len(self.seq_list)))

        #asser select idx contain all label
        label_set = set([self.label_list[idx] for idx in self.idx_list])
        if label_num == 60:
            assert max(label_set) == 59 and min(label_set) == 0 and len(label_set) == 60
        else:
            assert max(label_set) == 119 and min(label_set) == 0 and len(label_set) == 120

        import collections
        counter = collections.Counter([self.label_list[idx] for idx in self.idx_list])
        print('label frequency:', counter)


        print('{} {} data num: {}'.format(data_split, data_part, len(self.idx_list)))

    def slid_window(self, frame_num, w_size, stride, dilate):
        begin_idx = 0
        result = []
        idx_list = list(range(0, frame_num))
        while 1:
            end_idx = begin_idx + w_size * dilate
            #print(begin_idx, end_idx, begin_idx + (w_size - 1) * dilate)
            # if begin_idx + (w_size - 1) * dilate > frame_num-1:
            #     break
            tmp = idx_list[begin_idx:end_idx][::dilate]

            if len(tmp) < w_size:
                pad_num = w_size - len(tmp)
                tmp += [idx_list[-1]] * pad_num
                assert len(tmp) == self.w_size
                result.append(tmp)

            assert len(tmp) == w_size
            result.append(tmp)

            begin_idx += stride

            if tmp[-1] >= frame_num - w_size // 2 * dilate - 1 and begin_idx + (w_size - 1) * dilate > frame_num-1:
                #contain the last frame and overpass padd
                break


        return result


    def data_aug(self, skeleton):
        #skeleton: frame_num, 2, j_num, j_dim
        def scale(skeleton):
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            for person in skeleton:
                if person.sum() != 0:
                    person *= factor
            return skeleton

        def noise(skeleton):
            low = -0.2
            high = -low

            # random select 5 joints to add noise
            j_num = skeleton.shape[2]
            selected_joint = random.sample(list(range(j_num)), 5)
            assert len(selected_joint) == 5
            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for person in skeleton:
                    if person.sum() != 0:
                        #person: frame_num, joint_num, joint_dim
                        person[:, j_id] += noise_offset
            return skeleton


        def shear(skeleton, r=0.5):
            #shear: frame_num, 2, j_num, j_dim
            s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
            s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

            R = np.array([[1, s1_list[0], s2_list[0]],
                          [s1_list[1], 1, s2_list[1]],
                          [s1_list[2], s2_list[2], 1]])

            R = R.transpose()
            for p_id, person in enumerate(skeleton):
                if person.sum() != 0:
                    skeleton[p_id] = np.dot(person, R)
            return skeleton

        def time_interpolate(skeleton):

            skeleton = np.transpose(skeleton, [1, 0, 2, 3])
            assert skeleton.shape[1:] == (2, self.joint_num, 3)
            video_len = skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []
            for i in range(1, video_len):
                displace = skeleton[i] - skeleton[i - 1]  # d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i - 1] + displace)  # r*disp

            result.append(skeleton[-1])  # padding
            result = np.array(result)

            assert result.shape == skeleton.shape

            result = np.transpose(result, [1, 0, 2, 3])

            return result

        def time_crop(skeleton, ratio=0.2):

            skeleton = np.transpose(skeleton, [1, 0, 2, 3])
            assert skeleton.shape[1:] == (2, self.joint_num, 3)
            video_len = skeleton.shape[0]

            ratio = np.random.uniform(0, ratio)
            crop_frame = int(np.round(video_len * ratio))
            #print(skeleton.shape, ratio, crop_frame)

            if crop_frame > 0:
                begin_crop_frame = random.randint(0, crop_frame)
                end_crop_frame = crop_frame - begin_crop_frame

                #print(begin_crop_frame, end_crop_frame)

                skeleton = skeleton[begin_crop_frame:video_len - end_crop_frame]

                #print(result.shape)

                assert skeleton.shape[0] == video_len - crop_frame

            #print(result.shape)

            result = np.transpose(skeleton, [1, 0, 2, 3])

            return result


        # og_id = np.random.randint(3)
        skeleton = np.transpose(skeleton, [1, 0, 2, 3])  # 【frame_num, person_num, joint_num, joint_dim】 --> 【person_num, frame_num, joint_num, joint_dim】
        assert skeleton.shape[2:] == (25, 3) and skeleton.shape[0] == 2
        #print(skeleton.shape)
        #ag_id = randint(0, aug_num - 1)
        #aug_id_list = list(range(aug_num))
        #random.shuffle(aug_id_list)
        aug_id_list = [random.randint(0, 2), random.randint(3, 4)]
        #print(aug_id_list)
        #aug_id_list = [4]
        for aug_id in aug_id_list:
            #print(aug_id)
            if aug_id == 0:
                skeleton = scale(skeleton)
            elif aug_id == 1:
                skeleton = noise(skeleton)
            elif aug_id == 2:
                skeleton = shear(skeleton)
            elif aug_id == 3:
                skeleton = time_interpolate(skeleton)
            elif aug_id == 4:
                skeleton = time_crop(skeleton)

            else:
                raise ValueError('invalid augmenation type')


        skeleton = np.transpose(skeleton, [1, 0, 2, 3])
        #print(skeleton.shape)
        assert skeleton.shape[1:] == (2, self.joint_num, 3)

        return skeleton

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, ind):

        #print('ind', ind)
        ind = self.idx_list[ind]
        org_frame_num = self.video_info[ind]['frame_num']
        boby_num = self.video_info[ind]['boby_num']
        label = self.label_list[ind]

        joint_seq = copy.deepcopy(self.seq_list[ind])  # 3, max_frame, num_joint, max_body
        joint_seq = joint_seq.transpose(1, 3, 2, 0) #reshape

        joint_seq = joint_seq[:org_frame_num] #remove padding

        assert joint_seq.shape == (org_frame_num, 2, self.joint_num, 3)



        #pre_normalization:

        #data augmentation
        if self.use_data_aug and random.random() > 0.5:
            joint_seq = self.data_aug(joint_seq)
            assert joint_seq.shape[1:] == (2, self.joint_num, 3)
            assert joint_seq.shape[0] <= org_frame_num
            org_frame_num = joint_seq.shape[0]


        if self.view == 'joint':
            pass
        elif self.view == 'motion':
            #print(joint_seq[4,1, 10], joint_seq[5,1,10], joint_seq[6,1,10])
            motion = np.zeros_like(joint_seq) # [frame_num, joint_num, joint_dim]
            motion[1:, :, :] = joint_seq[1:, :, :] - joint_seq[:-1, :, :]
            joint_seq = motion
            #print(joint_seq[5, 1,10], joint_seq[6, 1, 10])

        elif self.view == 'bone':
            #print(joint_seq.shape)
            #print(joint_seq[33,1])
            bone = np.zeros_like(joint_seq) # [frame_num, joint_num, joint_dim]
            for v1, v2 in self.Bone:
                bone[:, :, v1 - 1] = joint_seq[:, :, v1 - 1] - joint_seq[:, :, v2 - 1]
            joint_seq = bone
            #print(joint_seq[33, 1])
            #print(bone.shape, joint_seq.shape)

        assert joint_seq.shape == (org_frame_num, 2, self.joint_num, self.joint_dim)


        #---sliding window

        #pad in the begining and end
        pad_num = self.w_size // 2 * self.dilate
        joint_seq = np.pad(joint_seq, [(pad_num, pad_num), (0, 0), (0, 0), (0, 0)], mode='constant')
        assert joint_seq.shape == (org_frame_num + pad_num*2, 2, self.joint_num, self.joint_dim)
        #use sliding window
        new_frame_num = joint_seq.shape[0]
        w_list = self.slid_window(frame_num=new_frame_num, w_size=self.w_size, stride=self.stride, dilate=self.dilate)
        assert w_list[-1][-1] >= org_frame_num + pad_num - 1
        #print(org_frame_num, w_list)
        joint_seq = [[joint_seq[idx] for idx in w] for w in w_list]
        joint_seq = np.array(joint_seq)
        assert joint_seq.shape == (len(w_list), self.w_size, 2, self.joint_num, self.joint_dim)

        w_num = len(joint_seq)
        w_pad_mask = np.zeros((w_num))

        sample = {'joint_seq': joint_seq, "label": label, 'w_pad_mask':w_pad_mask}

        return sample


def pad_joint_seq(seq_list):
    max_len = 0
    for seq in seq_list:
        max_len = max(seq.shape[0], max_len)

    _, w_size, body_num, j_num, j_dim = seq_list[0].shape
    assert body_num == 2 and (j_dim == 3 or j_dim == 9)

    result = np.zeros(((len(seq_list), max_len, w_size, body_num, j_num, j_dim)))

    for seq_idx, seq in enumerate(seq_list):
        w_num = seq.shape[0]
        assert w_num <= max_len
        result[seq_idx, :w_num] = seq
    result = np.array(result)
    return result

def autopad_mask(mask_list):

    #get max length
    max_len = 0
    for mask in mask_list:
        max_len = max(mask.shape[0], max_len)

    max_len = max_len #for information fusion token

    data_num = len(mask_list)

    result = np.ones((data_num, 2, max_len+1))

    result[:, 0, -1] = 0 #for information fusion token
    result[:, 1, -1] = 0

    for i in range(data_num):
        mask = mask_list[i]
        this_len = mask.shape[0]
        result[i, 0, :this_len] = mask
        result[i, 1, :this_len] = mask
    return result

def my_collate(batch):


    #joint seq list
    joint_seq = [item['joint_seq'] for item in batch]
    joint_seq = pad_joint_seq(joint_seq)
    joint_seq = torch.tensor(joint_seq) #to pytorch tensor

    #label list
    label = [item['label'] for item in batch]
    label = torch.tensor(label)


    w_pad_mask = [item['w_pad_mask'] for item in batch]
    w_pad_mask = autopad_mask(w_pad_mask)
    w_pad_mask = torch.tensor(w_pad_mask)

    assert joint_seq.shape[1] == w_pad_mask.shape[2] - 1


    return {'joint_seq': joint_seq, 'label':label, 'w_pad_mask':w_pad_mask}


if __name__ == "__main__":

    from tqdm import  tqdm

    dataset = NTU_Dataset(benchmark='xsub', data_split='val', w_size=5, stride=5, dilate=1, use_data_aug=True,
                          view='joint', data_part='all', label_num=60)

    for i in tqdm(range(len(dataset))):
        joint_seq = dataset[i]['joint_seq']




