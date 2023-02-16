# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB

import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
    
import json
import pickle
import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB

from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from PIL import Image
from pyquaternion import Quaternion
import pyquaternion

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# PoseNet
import torchvision.models as models
from torch import nn
import math
import torch.nn.functional as F
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result in json format')
    parser.add_argument(
        '--show-range',
        type=int,
        default=50,
        help='Range of visualization in BEV')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--vis-thred',
        type=float,
        default=0.3,
        help='Threshold the predicted results')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='../data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='video',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=20, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # load predicted results
    results = mmcv.load(args.res)['results']
    # load dataset information
    infos_path = args.root_path + '/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    infos = mmcv.load(infos_path)
    infos = infos['infos']
    
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    
    show_classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ]
        
    print('start visualizing results')
        
    for cnt, info in enumerate(infos[:min(args.vis_frames, len(infos))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(infos))))
        
        # Set cameras
        threshold = 0.35
        show_range = 60

        # Set figure size
        plt.figure(figsize=(21, 8))

        imsize = (1600, 900)
        box_vis_level = BoxVisibility.ANY

        for i, k in enumerate(views):
            # Draw camera views
            fig_idx = i + 1 if i < 3 else i + 1
            plt.subplot(2, 3, fig_idx)

            # Set camera attributes
            plt.title(k)
            plt.axis('off')
            plt.xlim(0, 1600)
            plt.ylim(900, 0)

            img = mmcv.imread(os.path.join(info['cams'][k]['data_path']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw images
            plt.imshow(img)

            """
            Box: global => ego => lidar => sensor => image?
              or global => ego => sensor => image?
            """
            ego2global_translation = info['cams'][k]['ego2global_translation']
            ego2global_rotation = info['cams'][k]['ego2global_rotation']

            lidar2ego_translation = info['lidar2ego_translation']
            lidar2ego_rotation =  Quaternion(info['ego2global_rotation'])

            sensor2ego_trans = info['cams'][k]['sensor2ego_translation']
            sensor2ego_rot = info['cams'][k]['sensor2ego_rotation']

            sensor2lidar_translation = info['cams'][k]['sensor2lidar_translation']
            sensor2lidar_rotation = info['cams'][k]['sensor2lidar_rotation']

            intrinsic = info['cams'][k]['cam_intrinsic']
            intrinsic = np.array(intrinsic)
            
            boxes_pred = []
            # for box_dict in list(results.values())[cnt]:
            for box_dict in results[info['token']]:
                if box_dict['detection_score'] >= threshold and box_dict['detection_name'] in show_classes:
                    box = Box(
                        box_dict['translation'],
                        box_dict['size'],
                        Quaternion(box_dict['rotation']),
                        name=box_dict['detection_name']
                    )

                    # box를 global => ego로 이동
                    box.translate(-np.array(ego2global_translation))
                    box.rotate(Quaternion(ego2global_rotation).inverse)

                    # box를 ego => lidar로 이동
                    # box.translate(np.array(lidar2ego_translation))
                    # box.rotate(Quaternion(lidar2ego_rotation).inverse)

                    # box를 lidar => camera로 이동
                    # box.translate(-np.array(sensor2lidar_translation))
                    # box.rotate(Quaternion(matrix=sensor2lidar_rotation).inverse)

                    # box를 ego => camera로 이동
                    box.translate(-np.array(sensor2ego_trans))
                    box.rotate(Quaternion(sensor2ego_rot).inverse)

                    if box_in_image(box, intrinsic, imsize, vis_level=box_vis_level):
                        c=cm.get_cmap('tab10')(show_classes.index(box.name))

                        # box를 camera => image로 이동해서 render
                        box.render(plt, view=intrinsic, normalize=True, colors=(c, c, c))

        # Set legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(),
                   by_label.keys(),
                   loc='upper right',
                   framealpha=1)

        plt.tight_layout(w_pad=0, h_pad=2)
        save_name ='output_%06d.jpg' % cnt
        plt.savefig(os.path.join("./vis/nusc", save_name))
        plt.close()

if __name__ == '__main__':
    main()
