import sys
sys.path.append('./')
from PIL import Image
import gradio as gr

from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


base_path = 'yisol/IDM-VTON'
example_path = os.path.join("/home/bohan/jiayi/IDM-VTON-train/gradio_demo/example/human/00034_00.jpg")


human_img = Image.open("/home/bohan/jiayi/IDM-VTON-train/gradio_demo/example/human/00034_00.jpg")


human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
# verbosity = getattr(args, "verbosity", None)
pose_img = args.func(args,human_img_arg)    
pose_img = pose_img[:,:,::-1]    
pose_img = Image.fromarray(pose_img).resize((768,1024))
pose_img.save('pose.jpg')
