import argparse
import torch
import os
import numpy as np
import pynvml
import time
import subprocess
import psutil
import datetime
import copy
import socket, errno
import shutil
from tqdm import tqdm
from contextlib import closing
import concurrent.futures
def copy_file(file_path, dst):
    # 判断文件是否存在
    if os.path.isfile(file_path):
        # 如果文件存在，那么复制文件
        shutil.copy(file_path, dst)

def copy_file_parallel(src, dst, process_num):
    # 获取文件夹下的所有文件
    file_list = os.listdir(src)
    # 计算文件总数
    file_num = len(file_list)
    # 创建一个进度条对象
    pbar = tqdm(total=file_num)
    # 创建一个进程池对象，指定进程数
    with concurrent.futures.ProcessPoolExecutor(max_workers=process_num) as executor:
        # 遍历文件列表
        for file_name in file_list:
            # 拼接文件的完整路径
            file_path = os.path.join(src, file_name)
            # 提交任务到进程池，传入复制文件的函数和参数
            executor.submit(copy_file, file_path, dst)
            # 更新进度条
            pbar.update(1)
    # 关闭进度条
    pbar.close()
def check_port_in_use(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return False
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            return True
        else:
            raise e
    finally:
        s.close()
def prepare_data_ssl(args):
    time_start = time.time()
    os.chdir(args.prepare_dir)
    # 将指令存入一个列表中
    commands = [
        # "cp ./data/kitti/semi_supervised_data_3dioumatch/scene_0.02/1/kitti_infos_train.pkl  ./data/kitti/",
        # "python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml",
        # "cp ./data/kitti/semi_supervised_data_3dioumatch/scene_0.02/1/kitti_infos_train_include_unlabel.pkl  ./data/kitti/",
        # "mv ./data/kitti/kitti_infos_train_include_unlabel.pkl ./data/kitti/kitti_infos_train.pkl"
        # "cp -r ./data/waymo/scene_40/80/waymo_processed_data_v0_5_0_test_v1_gt_database_train_sampled_1 ./data/waymo/waymo_processed_data_v0_5_0_test_v1_gt_database_train_sampled_1",
        # 顺序抽取数据
        "cp /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/scene_40/80/waymo_processed_data_v0_5_0_test_v1_infos_train.pkl /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test_v1_infos_train.pkl",
        "cp /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/scene_40/80/waymo_processed_data_v0_5_0_test_v1_infos_val.pkl /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test_v1_infos_val.pkl",
        "cp /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/scene_40/80/waymo_processed_data_v0_5_0_test_v1_waymo_dbinfos_train_sampled_1.pkl /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test_v1_waymo_dbinfos_train_sampled_1.pkl",
        # 随机抽取数据
        # "cp /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/20/10/waymo_processed_data_v0_5_0_test_v3_infos_train_ssl.pkl /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test_v3_infos_train.pkl",
        # "cp /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/20/10/waymo_processed_data_v0_5_0_test_v3_infos_val.pkl /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test_v3_infos_val.pkl",
        # "cp /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/20/10/waymo_processed_data_v0_5_0_test_v3_waymo_dbinfos_train_sampled_1.pkl /home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test_v3_waymo_dbinfos_train_sampled_1.pkl",
    ]   
    # 遍历指令列表
    for command in commands:
        os.system ("pwd")
        print(command)
        # 判断是否是cd指令
        if command.startswith ("cd"):
            # 改变工作目录
            os.chdir (command.split ()[1])
        else:
            # 执行指令
            os.system (command)
    # 复制文件 随机抽取数据
    # src_dir = '/home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/20/10/waymo_processed_data_v0_5_0_test_v3_gt_database_train_sampled_1'
    # dst_dir = '/home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test`_v3_gt_database_train_sampled_1'
    # 复制数据 顺序抽取数据
    src_dir = '/home/hswang/3D-det/my_model/Veritas_v1/data/waymo/semi_supervised_data_3dioumatch/scene_40/80/waymo_processed_data_v0_5_0_test_v1_gt_database_train_sampled_1'
    dst_dir = '/home/hswang/3D-det/my_model/Veritas_v1/data/waymo/waymo_processed_data_v0_5_0_test_v1_gt_database_train_sampled_1'
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    # 删除目标文件夹下的所有文件
    
    # 并行复制文件
    copy_file_parallel(src_dir, dst_dir, 16)
    time_end = time.time()
    print('prepare data time cost', time_end - time_start, 's')
    os.chdir(args.work_dir)

def get_any_commnd():
    # command = 'python preparedata/waymo/ego_info.py --data_folder /home/hswang/3D-det/ImmortalTracker/waymo --process 10'
    # command = 'python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset_debug.yaml --processed_data_tag waymo_processed_data_v0_5_0_debug'
    # command = ''
    # command = 'bash scripts/dist_train_debug.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR2/ckpts/unitr_pretrain.pth  --extra_tag self_and_cross_with_4d_dwconv_epo10_T2_mid_3f_notimeadded_decouple'
    # command = 'bash scripts/dist_train_debug.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR2/ckpts/unitr_pretrain_with_vmamba_B.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_3f_new_pretrain'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba2.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_allnew_pretrain'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba2.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_newgraphlss2'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_allnew'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_stage1_big'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba2_new_v2.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_stage123_full_newcoords_stage2h2_depthlss_newdownsample3030256_aug_no2'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba2_new_v3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_stage123_full_newcoords_stage2h2_newdownsample'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+newintelval.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR3/ckpts/unitr_pretrain_with_vmamba_B_with_voxelmamba_pretrained_v2.pth --extra_tag speed_test'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba2_new_v3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_stage123_full_newcoords_stage2h2'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full2.yaml --sync_bn --logger_iter_interval 100  --extra_tag self_and_cross_with_4d_dwconv_epo10_new_stage123_full_with2_stage2_win3030_g90_res_newsort_newcoords_no_downsample_no_pretrain_v2'
    command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full4.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_morevbackbone'
    command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full4_half.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_vaug_half'
    command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full4_stage32.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_vaug_stage3_down2'
    
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full4_lite.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_vaug_lite'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full9.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_lr001_sperate_trainval'
    
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full6.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_lr002_sperate'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full7.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_lr001'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full9_newpaste.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_lr0015_sperate_trainval_newpaste'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full8.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_lr001_sperate'
    # command = 'bash scripts/dist_train_debug.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full10.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_lr0015_sperateadamw2'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full4_noshift.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_noshift'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_with_depth'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full3_based.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_withdiff5_stage2times2'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_m`odels/unitr+lss+selfmamba3_full5_big.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_hign_resolusion'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2_alldataset'
    # command = 'bash scripts/dist_train_debug.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained_v2.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_backbone_fusion_v6'
    command = 'bash scripts/dist_train_debug.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr_map+lss.yaml --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --sync_bn --eval_map --logger_iter_interval 100 --extra_tag map_unitr_20p'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full5.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_abs_v2'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full2.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag base_version'
    # command = 'bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc_maskout'

    return command

def openpcdet_base_arser(parser):

    parser.add_argument('--mode', type=int, default=0, help='-1 for debug 0 for train 1 for val 2 for dataset 3 for ssl')
    # Add arguments for config files
    parser.add_argument('--config_file', type=str, default='cfgs/dsvt_models/dsvt_3D_1f_onestage_auto.yaml', help='Path to the config file for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    # Add arguments for extra tags
    parser.add_argument('--extra_tag', type=str, default=None, help='Extra tag')
    # Add arguments for other parameters
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--batch_size_per_gpu', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--logger_iter_interval', type=int, default=None, help='')
    return parser
# Define a function that takes command-line arguments and returns config files and other parameters
def get_config_files_and_params():
    # Create a parser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--work_dir', type=str, default='/home/hswang/3D-det/HSSDA_v1/HSSDA/tools/', help='Work directory')
    parser.add_argument('--debug', type=bool, default=False, help='Debug')
    parser.add_argument('--run_any_command', type=bool, default=False, help='Run any command')
    parser.add_argument('--prepare_dir', type=str, default='/home/hswang/3D-det/HSSDA_v1/HSSDA/', help='Prepare directory')
    parser.add_argument('--prepare_data', type=bool, default=False, help='Prepare data')
    parser.add_argument('--prepare_only', type=bool, default=False, help='Prepare only')
    parser.add_argument('--can_gpu_list', type=str, default='7,6,5,4,3,2,1,0', help='List of available GPUs')
    parser.add_argument('--time_sum', type=int, default=0, help='Sum of time')
    parser.add_argument('--min_gpu', type=int, default=1, help='Minimum number of GPUs')
    parser.add_argument('--min_gpu_threshold', type=float, default=0.2, help='Minimum GPU threshold')
    parser.add_argument('--max_gpu', type=int, default=4, help='Maximum number of GPUs')
    parser.add_argument('--time_step', type=int, default=10, help='Time step')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master address')
    parser.add_argument('--master_port', type=int, default=23456, help='Master port')
    parser.add_argument('--log_dir', type=str, default='./my_run_log.txt', help='Log directory')
    parser.add_argument('--watch', type=bool, default=False, help='Watch GPU')
    parser.add_argument('--run_with_subprocess', type=bool, default=False, help='Run with subprocess')
    parser.add_argument('--subprocess_check_interval', type=int, default=10, help='Subprocess check interval')
    parser.add_argument('--print_watch', type=bool, default=False, help='Print watch')
    parser.add_argument('--run_after_port', type=int, default=None, help='Run after port')
    
    # Call the function to parse the arguments
    parser = openpcdet_base_arser(parser)
    # Parse the arguments
    args = parser.parse_args()

    args.can_gpu_list = [int(i) for i in args.can_gpu_list.split(',')]
    # Return the arguments as variables
    return args

def add_args(args, app_args):
    # app_args += ('--logger_iter_interval 100 ')
    if args.logger_iter_interval is not None:
        app_args += ('--logger_iter_interval {} '.format(args.logger_iter_interval))
    if args.pretrained_model is not None:
        app_args += ' --pretrained_model {} '.format(args.pretrained_model)
    if args.sync_bn:
        app_args += ' --sync_bn '
    return app_args

def get_command(args, gpu_list = []):
    if args.run_any_command:
        return get_any_commnd()
    # import pdb; pdb.set_trace()
    assert args.extra_tag is not None, 'extra_tag is None'
    batch_size_total = args.batch_size_per_gpu * len(gpu_list) if args.batch_size_per_gpu is not None else args.batch_size
    app_args = '--extra_tag {}  --batch_size {} --workers {} '.format(args.extra_tag, batch_size_total, args.workers)
    if args.mode == 0: # train
        app_args += ''
    elif args.mode == 1: # val
        if args.ckpt is not None:
            app_args += ' --ckpt {} '.format(args.ckpt)
        else:
            app_args += ' --eval_all'
    else:
        assert 0, 'mode error'
    app_args = add_args(args, app_args)
    
    base_command = 'python -m torch.distributed.launch --nproc_per_node={} --master_addr {} --master_port {} {}.py --cfg_file {} --launcher pytorch'.format(len(gpu_list), args.master_addr, args.master_port,'test' if args.mode == 1 else 'train', args.config_file)    
    if args.debug:
        debug_command = 'CUDA_LAUNCH_BLOCKING=1 '
        base_command = debug_command + base_command
    return base_command + ' ' + app_args
def get_gpu_list(args, max_gpu = None):
    if max_gpu is None:
        max_gpu = args.max_gpu
    pynvml.nvmlInit()
    gpu_list = []
    for i in args.can_gpu_list:

        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.used / (meminfo.used + meminfo.free))
        if meminfo.used / (meminfo.used + meminfo.free) < args.min_gpu_threshold:
            gpu_list.append(i)
            if(len(gpu_list) >= max_gpu):
                break
    return gpu_list
def log_info(args,cammand, gpu_list, time_sum, log):
    print('time_sum: {}s, {} of gpu avaliable'.format(time_sum, len(gpu_list)))
    print('====================================================================================================',file=log)
    print('time_sum: {}s, {} of gpu avaliable'.format(time_sum, len(gpu_list)),file=log)
    ms = round(time.time() * 1000)
    print(time.strftime("%Y-%m-%d %H:%M:%S.", time.localtime(ms // 1000)) + str(ms % 1000),file=log)
    print(os.environ["PATH"],file=log)
    print('time_sum: {}s, {} of gpu avaliable'.format(time_sum, len(gpu_list)),file=log)

    print('CUDA_VISIBLE_DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'],file=log)
    
    print(cammand, file=log)
    print(cammand)

def wait_for_command(args):
    flag = True
    time_sum = 0
    while True:
        gpu_list = get_gpu_list(args)
        port_used = check_port_in_use(args.master_addr, args.master_port)
        print('time_sum: {}s, {} of {} gpu {} avaliable, port_used: {}'.format(time_sum, len(gpu_list), len(args.can_gpu_list),args.can_gpu_list, port_used))
        print(get_command(args, gpu_list))
        
        if len(gpu_list) >= args.min_gpu and not port_used:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_list])
            command = get_command(args, gpu_list)
            break
        time_sum += args.time_step
        time.sleep(args.time_step) 
    return command, gpu_list, time_sum
def port_is_used(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((host, port)) == 0:
            print("Port is open")
        else:
            print("Port is not open")
def run_main(args):
    if args.prepare_data and args.prepare_only:
        prepare_data_ssl(args)
        return
    os.chdir(args.work_dir)
    log = open(args.log_dir, mode = "a+", encoding = "utf-8")
    if args.run_after_port is not None:
        # 每隔10s查看指定端口是否有程序运行
        while True:
            if not port_is_used(args.master_addr,args.run_after_port):
                break
            time.sleep(10)
            print('wait for port {} to be free'.format(args.run_after_port))
    command, gpu_list, time_sum = wait_for_command(args)
    
    if args.prepare_data:
        prepare_data_ssl(args)
    log_info(args, command, gpu_list, time_sum, log)
    if args.watch:
        watch_cammand = 'exec python my_watch.py --watch_gpu_list {}'.format(','.join([str(i) for i in gpu_list]))
        watch_process = subprocess.Popen(watch_cammand, shell=True)

    os.system(command)
    # cascade_command = 'bash scripts/dist_train_debug2.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr+lss+selfmamba3_full3.yaml --sync_bn --logger_iter_interval 100 --pretrained_model /home/hswang/AD/Fusion/UniTR4/ckpts/unitr_pretrain_with_vmamba_B_with_lionmamba_pretrained.pth --extra_tag self_and_cross_with_4d_dwconv_epo10_new_full_with_multi_lidar_res_lidar_all_acc2_lsslite_with_depth_v2'
    # time.sleep(600)
    # os.system(cascade_command)
    if args.watch:
        watch_process.kill()

if __name__ == '__main__':
    args = get_config_files_and_params()
    run_main(args)
    

    
    
    
    