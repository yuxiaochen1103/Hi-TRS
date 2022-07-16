#split data with multiple persons to single person
#cut max_length to 200
import os
import json
import numpy as np
from numpy.lib.format import open_memmap
import pickle
from tqdm import tqdm

def read_info(pth):
    src_file = open(pth)
    data = json.load(src_file)
    return data

def read_joint_seq(pth):
    joint_seq = np.load(pth, mmap_mode='r') # data_num, 3, max_frame, num_joint, max_body
    return joint_seq



benchmark = ['xsub', 'xview']
part = ['train', 'val']

#benchmark = ['xsub']
#part = ['pre_train']

base_fold = './data/NTU_60'

for b in benchmark:
    for p in part:
        #print(b, p)
        #load video info list
        info_pth = os.path.join(base_fold, '{}/{}_video_info.json'.format(b, p))
        video_info = read_info(info_pth)
        # load joint sequence list
        seq_pth = os.path.join(base_fold, '{}/{}_data.npy'.format(b, p))
        joint_seq = read_joint_seq(seq_pth) # data_num, 3, max_frame, num_joint, max_body
        assert joint_seq.shape[0] == len(video_info)
        print('joint_seq shape:', joint_seq.shape)

        video_num = len(video_info)
        seq_num = 0

        #get seq_um
        for v_id in range(video_num):
            info = video_info[v_id]
            body_num = info['boby_num']
            seq_num += body_num

        print('seq num:', seq_num)

        max_frame = 300
        num_joint = 25


        res_pth = base_fold + '/{}/{}_single_person.npy'.format(b, p)
        res_array = open_memmap(
            res_pth,
            dtype='float32',
            mode='w+',
            shape=(seq_num, max_frame, num_joint, 3))

        seq_id = 0
        f_num_list = []
        for v_id in tqdm(range(video_num)):
            info = video_info[v_id]
            boby_num = info['boby_num']
            frame_num = info['frame_num']
            # data_num, 3, max_frame, num_joint, max_body
            for bid in range(boby_num):
                #[3, max_frame, num_joint]
                t = joint_seq[v_id, :, :, :, bid] #[3, max_frame, num_joint]



                #print(t.shape)
                t = np.transpose(t, (1, 2, 0))
                #print(t.shape)
                #print(t.shape)
                res_array[seq_id] = t
                seq_id += 1
                f_num_list.append(frame_num)


        assert seq_id == seq_num
        assert len(f_num_list) == seq_num

        #save length list
        res_pth = os.path.join(base_fold, '{}/{}_fnum_single_person.pkl'.format(b, p))
        with open(res_pth, "wb") as fp:
            pickle.dump(f_num_list, fp)





