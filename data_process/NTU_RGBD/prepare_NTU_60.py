import os
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import json

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    boby_num = 0
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                #print(j,v)
                if m < max_body and j < num_joint:
                    boby_num = max(boby_num, m + 1)
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data, boby_num


def read_color_xy(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((2, seq_info['numFrame'], num_joint, max_body))
    boby_num = 0
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                #print(j,v)
                if m < max_body and j < num_joint:
                    boby_num = max(boby_num, m + 1)
                    data[:, n, j, m] = [v['colorX'], v['colorY']]
                else:
                    pass
    return data, boby_num


def gendata(skl_gt_fold,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):

    #get list for unvalid video
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    #get data for the specific part
    sample_name = []
    sample_label = []
    for filename in os.listdir(skl_gt_fold):
        if filename in ignored_samples:
            continue

        #parse label
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        #split train and test
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp_xyz = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))
    video_info_list = []
    for i, s in enumerate(tqdm(sample_name)):

        data, boby_num = read_xyz(
            os.path.join(skl_gt_fold, s), max_body=max_body, num_joint=num_joint)


        fp_xyz[i, :, 0:data.shape[1], :, :] = data

        frame_num = data.shape[1]
        video_info_list.append({'frame_num':frame_num, 'boby_num':boby_num})

    res_pth = os.path.join(out_path, '{}_video_info.json'.format(part))
    #
    with open(res_pth, 'w') as res_file:
        json.dump(video_info_list, res_file)



if __name__ == '__main__':

    training_subjects = [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
    ]
    training_cameras = [2, 3]
    max_body = 2
    num_joint = 25
    max_frame = 300
    toolbar_width = 30

    base_fold = './data'
    skl_gt_fold = os.path.join(base_fold, 'nturgb+d_skeletons')
    ignored_sample_path = os.path.join(base_fold, 'samples_with_missing_skeletons.txt')
    out_fold = os.path.join(base_fold, 'NTU_60')


    benchmark = ['xsub', 'xview']
    part = ['train', 'val']

    for b in benchmark:
        print(b)
        for p in part:
            print(p)
            out_path = os.path.join(out_fold, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                skl_gt_fold,
                out_path,
                ignored_sample_path,
                benchmark=b,
                part=p)
