from torch.utils.data import Dataset
import torch
import numpy as np
import os.path as osp
import random
import pickle
import copy


class NTU_Pretrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, w_size, stride, dilate,
                 benchmark, view, label_num,
                 data_split, use_data_aug):

        """
        Args:
            w_size: window size
            stride: sliding window stride
            label_num: 14/28 using 14 setting or 18 setting
            test_sub_id: id of the subject used for training or testing
            data_split: train or test
            use_data_aug: flag for using data augmentation
        """

        #hd5py

        #--numpy input
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

        print('loading NTU_{} data......'.format(label_num))
        print('benchmark:', benchmark, 'data_split:', data_split, 'view:', self.view)

        print('use_data_aug: ', self.use_data_aug)


        # --numpy input
        # #load seq_list
        # print('loading sequence list from disk')
        seq_pth = osp.join(self.data_fold, '{}/{}_single_person.npy'.format(benchmark, data_split))
        self.seq_list = np.load(seq_pth, mmap_mode='r') #

        # load length list
        # print('loading frame_num list from disk')
        len_pth = osp.join(self.data_fold, '{}/{}_fnum_single_person.pkl'.format(benchmark, data_split))
        with open(len_pth, "rb") as fp:  # Unpickling
            self.len_list = pickle.load(fp)

        assert len(self.seq_list) == len(self.len_list)

        print('org data num:',len(self.len_list))
        #remove the seqs whose length is smaller than 10
        print('filter out invalid data......')
        self.valid_idx = []
        for idx, ele in enumerate(self.len_list):
            if ele > (2 * stride / 0.8) + 1: #contains at least 3 windows after cropping
                self.valid_idx.append(idx)

        assert self.seq_list.shape[-1] == 3 and self.seq_list.shape[-2] == self.joint_num

        print('data num:', len(self.valid_idx))



    def slid_window(self, frame_num, w_size, stride, dilate):
        begin_idx = 0
        result = []
        idx_list = list(range(0, frame_num))
        while 1:
            # if :
            #     break

            end_idx = begin_idx + w_size * dilate
            #print(begin_idx, end_idx, begin_idx + (w_size - 1) * dilate)
            tmp = idx_list[begin_idx:end_idx][::dilate]

            if len(tmp) < w_size:
                pad_num = w_size - len(tmp)
                tmp += [idx_list[-1]] * pad_num
                assert len(tmp) == self.w_size
                #result.append(tmp)

            assert len(tmp) == w_size
            result.append(tmp)

            #print(tmp)
            #print(result)

            begin_idx += stride

            #print(tmp[-1] >= frame_num - w_size // 2 * dilate - 1)
            if tmp[-1] >= frame_num - w_size // 2 * dilate - 1 and begin_idx + (w_size - 1) * dilate > frame_num-1:
                #contain the last frame and overpass padd
                break

            #print(begin_idx, end_idx)
            #if end_idx > :
                #break

        #print(result)
        return result

    def random_mask(self, joint_seq):

        masked_seq = np.zeros(joint_seq.shape)
        mask = np.zeros((joint_seq.shape[0], joint_seq.shape[1], 1))


        f_num, joint_num, joint_dim = masked_seq.shape

        #mask_token = [1 for i in range(joint_dim)]
        mask_token = np.ones(joint_dim)


        for f_id in range(f_num):
            for j_id in range(joint_num):
                prob = random.random()
                if prob < 0.15:
                    mask[f_id, j_id] = 1
                    prob = random.random()
                    if prob < 0.8: #Randomly 80% of tokens, gonna be a [MASK] token
                        masked_seq[f_id, j_id, :] = mask_token
                    elif 0.8 <= prob < 0.9: #Randomly 10% of tokens, gonna be a another joint
                        #sample frame idx
                        f_id_list = list(range(f_num))
                        f_id_list.remove(f_id)
                        s_f_id = random.choice(f_id_list)

                        #sample joint idx
                        j_id_list = list(range(joint_num))
                        j_id_list.remove(j_id)
                        s_j_id = random.choice(j_id_list)

                        masked_seq[f_id, j_id, :] = joint_seq[s_f_id, s_j_id, :]
                    else:
                        masked_seq[f_id, j_id] = joint_seq[f_id, j_id]
                else:
                    masked_seq[f_id, j_id] = joint_seq[f_id, j_id]

        return masked_seq, mask

    def data_aug(self, skeleton):
        #skeleton: frame_num, j_num, j_dim
        def scale(skeleton):#
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            skeleton *= factor
            return skeleton

        def noise(skeleton):
            low = -0.2
            high = -low

            # random select 5 joints to add noise
            j_num = skeleton.shape[1]
            selected_joint = random.sample(list(range(j_num)), 5)
            assert len(selected_joint) == 5
            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                skeleton[:, j_id] += noise_offset
            return skeleton


        def shear(skeleton, r=0.5):
            #shear: frame_num, 2, j_num, j_dim
            s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
            s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

            R = np.array([[1, s1_list[0], s2_list[0]],
                          [s1_list[1], 1, s2_list[1]],
                          [s1_list[2], s2_list[2], 1]])

            R = R.transpose()
            skeleton = np.dot(skeleton, R)
            return skeleton

        def time_interpolate(skeleton):

            assert skeleton.shape[1:] == (self.joint_num, 3)
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

            return result

        def time_crop(skeleton, ratio=0.2):

            assert skeleton.shape[1:] == (self.joint_num, 3)
            video_len = skeleton.shape[0]

            ratio = np.random.uniform(0, ratio)
            #ratio = 0.2
            crop_frame = int(np.round(video_len * ratio))
            #print(skeleton.shape, ratio, crop_frame)

            if crop_frame > 0:
                begin_crop_frame = random.randint(0, crop_frame)
                end_crop_frame = crop_frame - begin_crop_frame

                #print(begin_crop_frame, end_crop_frame)

                skeleton = skeleton[begin_crop_frame:video_len - end_crop_frame]

                #print(result.shape)

                assert skeleton.shape[0] == video_len - crop_frame

            return skeleton


        # og_id = np.random.randint(3)
        assert skeleton.shape[1:] == (25, 3) and len(skeleton.shape) == 3
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

        assert skeleton.shape[1:] == (25, 3) and len(skeleton.shape) == 3

        return skeleton


    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, ind):

        ind = self.valid_idx[ind]
        joint_seq = copy.deepcopy(self.seq_list[ind]) #frame_num, 25, 3
        org_frame_num = self.len_list[ind]

        joint_seq = joint_seq[:org_frame_num, :, :] #remove padding

        #use data augmentation
        if self.use_data_aug and random.random() < 0.5:
            joint_seq = self.data_aug(joint_seq)
            assert joint_seq.shape[1:] == (self.joint_num, 3)
            assert joint_seq.shape[0] <= org_frame_num
            org_frame_num = joint_seq.shape[0]



        if self.view == 'joint':
            pass
        elif self.view == 'motion':
            #print(joint_seq[4,10], joint_seq[5,10], joint_seq[6,10])
            motion = np.zeros_like(joint_seq) # [frame_num, joint_num, joint_dim]
            motion[1:, :, :] = joint_seq[1:, :, :] - joint_seq[:-1, :, :]
            joint_seq = motion
            #print(joint_seq[5, 10], joint_seq[6, 10])

        elif self.view == 'bone':
            #print(joint_seq.shape)
            bone = np.zeros_like(joint_seq) # [frame_num, joint_num, joint_dim]
            for v1, v2 in self.Bone:
                bone[:, v1 - 1] = joint_seq[:, v1 - 1] - joint_seq[:, v2 - 1]
            joint_seq = bone
            #print(bone.shape, joint_seq.shape)

        assert joint_seq.shape == (org_frame_num, self.joint_num, self.joint_dim)



        # **************************
        # *******frame level*******
        # **************************
        #random mask
        masked_seq, mask_seq_mask = self.random_mask(joint_seq)

        # **************************
        # *******clip level*******
        # **************************
        #calculate frame-level optical flow
        frame_of_list = [0]
        for i in range(1, org_frame_num):
            of = abs(joint_seq[i] - joint_seq[i - 1]).sum()
            frame_of_list.append(of)
        frame_of_list = np.array(frame_of_list)
        assert len(frame_of_list) == org_frame_num

        #pad the seq
        pad_num = self.w_size // 2 * self.dilate
        #masked_seq: frame_num, joint_num, 2
        joint_seq = np.pad(joint_seq, [(pad_num, pad_num), (0, 0), (0, 0)], mode='constant')
        masked_seq = np.pad(masked_seq, [(pad_num, pad_num), (0, 0), (0, 0)], mode='constant')
        mask_seq_mask = np.pad(mask_seq_mask, [(pad_num, pad_num), (0, 0), (0, 0)], mode='constant')
        frame_of_list = np.pad(frame_of_list, [(pad_num, pad_num)], mode='constant')
        new_frame_num = org_frame_num + 2 * pad_num
        assert joint_seq.shape[:2] == masked_seq.shape[:2] == mask_seq_mask.shape[:2] == (new_frame_num, self.joint_num)

        #----cal clip_level optical flow
        # the first and padded frames is not counted to calculate mean frame
        valid_of_mask = np.ones(joint_seq.shape[0])
        valid_of_mask[:pad_num+1] = 0
        valid_of_mask[-pad_num:] = 0
        #get all posible clip
        clip_list = self.slid_window(frame_num=new_frame_num, w_size=self.w_size, stride=1, dilate=self.dilate)
        #print(new_frame_num, org_frame_num, clip_list)
        assert len(clip_list) == org_frame_num
        clip_of_list = [frame_of_list[clip].sum() / valid_of_mask[clip].sum() for clip in clip_list]

        #sample a clip that has large movement
        valid_idx_list = [clip_idx for clip_idx, of in enumerate(clip_of_list) if of > 0]
        #print(len(clip_of_list), len(valid_idx_list))

        if len(valid_idx_list) > 0:
            #print(len(valid_idx_list))
            tgt_clip_idx = random.sample(valid_idx_list, 1)[0]
            c_motion_valid_flg = 1
        else:
            tgt_clip_idx = 0
            c_motion_valid_flg = 0  # flag indicate whether the tgt clip is valid


        #random swap two frame to swap
        while 1:
            swap_loc_1 = random.randint(0, self.w_size-1)
            swap_loc_2 = random.randint(0, self.w_size-1)
            if abs(swap_loc_1 - swap_loc_2) >= 2:
                frame_permute_idx = list(range(self.w_size))
                frame_permute_idx[swap_loc_1] = swap_loc_2
                frame_permute_idx[swap_loc_2] = swap_loc_1
                break

        org_clip = clip_list[tgt_clip_idx]
        assert len(org_clip) == len(frame_permute_idx)
        permute_clip = [org_clip[idx] for idx in frame_permute_idx] #each element is frame_idx
        c_motion_clip = [org_clip, permute_clip]


        #sliding window
        # **************************
        # *******video level*******
        # **************************
        assert new_frame_num == joint_seq.shape[0]

        w_list = self.slid_window(frame_num=new_frame_num, w_size=self.w_size, stride=self.stride, dilate=self.dilate)

        assert len(w_list) >= 3
        last_c = w_list[-1] #last clip is used for future prediction
        w_list = w_list[:-1]

        w_num = len(w_list) #the last clip is used for prediction
        w_pad_mask = np.zeros(w_num)


        assert last_c[-1] >= org_frame_num-1 #assert the last window contains last frame


        if w_num == 2:
            permute_v = [1, 0]
        else:
            while 1:
                swap_loc_1 = random.randint(0, w_num-1)
                swap_loc_2 = random.randint(0, w_num-1)
                if abs(swap_loc_1 - swap_loc_2) >= 2:
                    if 1:
                        permute_v = list(range(w_num))
                        permute_v[swap_loc_1] = swap_loc_2
                        permute_v[swap_loc_2] = swap_loc_1
                        break

        assert len(w_list) == len(permute_v)



        sample = {'masked_seq': masked_seq, "org_seq": joint_seq, 'mask_seq_mask': mask_seq_mask,
                  'c_motion_clip': c_motion_clip, 'c_motion_valid_flg': c_motion_valid_flg,
                  'permute_v': permute_v,
                  'w_list': w_list, 'w_pad_mask': w_pad_mask, 'last_c':last_c}

        return sample


def pad_joint_seq(seq_list):
    max_len = 0
    for seq in seq_list:
        max_len = max(seq.shape[0], max_len)

    joint_num, dim = seq.shape[1:]

    result = np.zeros((len(seq_list), max_len, joint_num, dim))
    for seq_idx, seq in enumerate(seq_list):
        frame_num, joint_num, dim = seq.shape
        #print(seq.shape)
        assert frame_num <= max_len

        result[seq_idx, :frame_num] = seq

    return result


def get_longest_w(w_list):
    #w_list: a list of slid_window information
    #w: w_num, w_size

    max_len = 0
    max_w = None
    for w in w_list:
        w_len = len(w)
        if w_len > max_len:
            max_len = w_len
            max_w = w

    #assert
    for w in w_list:
        w_len = len(w)
        #print(w, max_w)
        assert w[:-1] + max_w[w_len-1:] == max_w

    return max_w

def autopad_mask(mask_list):

    #get max length
    max_len = 0
    for mask in mask_list:
        max_len = max(len(mask), max_len)

    max_len = max_len + 1 #+1 for information fusion token

    data_num = len(mask_list)

    result = np.ones((data_num, max_len))

    result[:,-1] = 0 #for information fusion token

    for i in range(data_num):
        mask = mask_list[i]
        this_len = mask.shape[0]
        result[i, :this_len] = mask
    return result


def pad_permute_idx(permute_idx_list):

    max_len = 0
    for idx_list in permute_idx_list:
        max_len = max(len(idx_list), max_len)

    for idx_list in permute_idx_list:
        w_num = len(idx_list)
        if w_num < max_len:
            pad_seq = list(range(w_num, max_len))
            idx_list += pad_seq
        assert len(idx_list) == max_len

    return np.array(permute_idx_list)

def my_collate(batch):


    # batch contains a list of tuples of structure (sequence, target)
    # **********
    #-----frame-level
    # **********
    masked_seq = [item['masked_seq'] for item in batch]
    masked_seq = pad_joint_seq(masked_seq)
    masked_seq = torch.tensor(masked_seq)

    org_seq = [item['org_seq'] for item in batch]
    org_seq = pad_joint_seq(org_seq)
    org_seq = torch.tensor(org_seq)

    mask_seq_mask = [item['mask_seq_mask'] for item in batch]
    mask_seq_mask = pad_joint_seq(mask_seq_mask)
    mask_seq_mask = torch.tensor(mask_seq_mask)

    assert  masked_seq.shape[:3] == org_seq.shape[:3] == mask_seq_mask.shape[:3]

    # **********
    # clip level
    # **********

    #frame permure
    c_motion_clip = [item['c_motion_clip'] for item in batch]
    c_motion_clip = torch.tensor(c_motion_clip)  # used for clip_level order prediction problem

    c_motion_valid_flg = [item['c_motion_valid_flg'] for item in batch]
    c_motion_valid_flg = torch.tensor(c_motion_valid_flg)  # used for clip_level order prediction problem



    #**********
    #video level
    # **********
    w_list = [item['w_list'] for item in batch]
    slid_window = get_longest_w(w_list)
    assert slid_window[-1][-1] < org_seq.shape[1]


    w_pad_mask = [item['w_pad_mask'] for item in batch]
    w_pad_mask = autopad_mask(w_pad_mask)
    w_pad_mask = torch.tensor(w_pad_mask)

    last_c = [item['last_c'] for item in batch]
    last_c = torch.tensor(last_c)

    permute_v = [item['permute_v'] for item in batch]
    permute_v = pad_permute_idx(permute_v)
    permute_v = torch.tensor(permute_v)


    #print(masked_seq.shape, org_seq.shape, mask_seq_mask.shape)
    assert permute_v.shape[1] == len(slid_window) == w_pad_mask.shape[1] - 1



    return {'masked_seq': masked_seq, "org_seq": org_seq, 'mask_seq_mask': mask_seq_mask,
            'c_motion_clip':c_motion_clip, 'c_motion_valid_flg': c_motion_valid_flg,
            'permute_v': permute_v,
            'slid_window': slid_window, 'w_pad_mask': w_pad_mask,
            'last_c':last_c}


if __name__ == "__main__":

    from tqdm import  tqdm


    dataset = NTU_Pretrain(
                           w_size=7, stride=7, dilate=2,
                           benchmark='xview', label_num=60, view='joint',
                           use_data_aug=True, data_split='train'
                           )


    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        collate_fn=my_collate,
        pin_memory=True,
        shuffle=True)


    #i = 9674
    max_seq = 0
    for i, data in enumerate(tqdm(train_loader)):
        1


    print(max_seq)




