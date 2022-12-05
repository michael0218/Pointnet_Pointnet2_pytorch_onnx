"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y 
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal) 
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()


        dummy_point = torch.randn(24, 6, 2048)
        dummy_classes = torch.randn(24, 1, 16)
        torch.onnx.export(classifier,         # model being run 
            (dummy_point, dummy_classes),       # model input (or a tuple for multiple inputs) 
            "part_seg_opset11.onnx",       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput']) 
        print(" ") 
        print('Model has been converted to ONNX')


        torch.onnx.export(classifier,         # model being run 
            (dummy_point, dummy_classes),       # model input (or a tuple for multiple inputs) 
            "part_seg_opset12.onnx",       # where to save the model  
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=12,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput']) 
        print(" ") 
        print('Model has been converted to ONNX')

if __name__ == '__main__':
    args = parse_args()
    main(args)
