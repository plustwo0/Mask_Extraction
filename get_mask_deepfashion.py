import argparse
import sys
sys.path.append('./')

import torch
import os
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

import numpy as np
import cv2
from PIL import Image, ImageDraw

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384,height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]
        

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

parsing_model = Parsing(0)
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(device)

def get_mask(img_path):
    human_img = Image.open(img_path)
    human_img = human_img.resize((750,1101)) # [768, 1024] is recommended
    keypoints,location = openpose_model(human_img.resize((384,512)))
    model_parse, _ = parsing_model(human_img.resize((384,512)))
    mask, mask_gray = get_mask_location('hd', location, model_parse, keypoints)
    mask = mask.resize((750,1101)) # [768, 1024] is recommended
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((750,1101)) # [768, 1024] is recommended

    return mask, mask_gray, pose_img

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--root_path",type=str,default="test_data/full_body",required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # root_path = "test_data/full_body"
    # root_path = "test_data/lower_body"
    # root_path = "test_data/upper_body"
    for dir_path, _, filenames in os.walk(args.root_path):
        pic_path = dir_path
        mask_path = dir_path
        ano_path = dir_path
        pose_path = dir_path
        
        for img in filenames:
            # if 'segment' in img or 'flat' in img or 'densepose' in img or 'mask' in img or 'inpaint' in img or '.jpg' not in img:
            #     continue
            pic = os.path.join(pic_path, img)
            msk_img = img.split('.')[0] + '_mask.jpg'
            msk = os.path.join(mask_path, msk_img)
            ano_img = img.split('.')[0] + '_inpaint.jpg'
            ano = os.path.join(ano_path, ano_img)
            densepose_img = img.split('.')[0] + '_densepose.jpg'
            densepose = os.path.join(pose_path, densepose_img)
            try:
                mask, mask_gray,pose_img = get_mask(pic)
                mask.save(msk)
                mask_gray.save(ano)
                pose_img.save(densepose)
            except:
                print(dir_path+img)
            
       
