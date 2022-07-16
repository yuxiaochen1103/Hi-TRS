########## The codes are to reproduce our reported results on the NTU-60 dataset.

########## Prerequisites:
python-3.8
pytorch-1.7.0
pyyaml
numpy
tensorboardx

########## Data Preparation
1. Download the NTU-60 3D skeleton data from its official website, and unzip the downloaded data to './data'.
2. run 'python data_process/NTU_RGBD/prepare_NTU_60.py' to pre-process the data for action recognition.
3. run 'python data_process/NTU_RGBD/split_to_single_person.py' to pre-process the data for pre-training.

########## Pre-training
1. run " CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_on_NTU.py --cfg_pth 'cfg/NTU/pretrain/cfg.yaml' " to pre-train the Hi-TRS
(our pre-trained checkpoint under the xsub setting is provided at './checkpoint')

########## Fine-tuning
1. run "CUDA_VISIBLE_DEVICES=0,1 python train_on_NTU_AR.py --ft_cfg_pth 'cfg/NTU/AR/cfg.yaml'" to fine-tune the pre-trained Hi-TRS for action recognition under the supervised setting.
2. set the data_part in 'cfg/NTU/AR/cfg.yaml' file to 'ten' (10%), 'five' (5%) and 'one' (1%) and run the previous command to conduct the experiments the semi-supervised setting.
3. set the linear, data_part, lr as 1, 'all', 0.01 in the 'cfg/NTU/AR/cfg.yaml' to run our experiments under the linear setting.
