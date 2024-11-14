

## Requirements

```
pip install -r requirements.txt
```

## Data preparation
1. Download [model_final_162be9.pkl](https://huggingface.co/yisol/IDM-VTON/tree/main/densepose) and put it under `ckpt/densepose/`.
2. Download [parsing_atr.onnx](https://huggingface.co/yisol/IDM-VTON/tree/main/humanparsing) and [parsing_lip.onnx](https://huggingface.co/yisol/IDM-VTON/tree/main/humanparsing) and put it under `ckpt/humanparsing/`.
3. Download [body_pose_model.pth](https://huggingface.co/yisol/IDM-VTON/tree/main/openpose/ckpts) and put it under `ckpt/openpose/ckpts`.

```
python get_mask_deepfashion.py --root_path test_data/full_body/
```

or

```
python get_mask_deepfashion.py --root_path path_to_your_data
```

## Acknowledgements

Thanks [IDM-VTION](https://github.com/yisol/IDM-VTON) and [IDM-VITON-train](https://github.com/luxiaolili/IDM-VTON-train) for most codes.

Thanks [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for human segmentation.

Thanks [Densepose](https://github.com/facebookresearch/DensePose) for human densepose.


## License
The codes and checkpoints in this repository are under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


