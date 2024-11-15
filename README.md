# CDG
# [Submit PR 2024] Concept-Guided Domain Generalization for Semantic Segmentation


## Environment Setup
To set up your environment, execute the following commands:
```bash
conda create -n CDG -y
conda activate CDG
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20' # optional for DINOv2
pip install -r requirements.txt
pip install future tensorboard
```

## Dataset Preparation
The Preparation is similar as [Rein](https://github.com/w1oves/Rein).

**Cityscapes:** Download `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from [Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/) and extract them to `data/cityscapes`.

**Mapillary:** Download MAPILLARY v1.2 from [Mapillary Research](https://research.mapillary.com/) and extract it to `data/mapillary`.

**GTA:** Download all image and label packages from [TU Darmstadt](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `data/gta`.

Prepare datasets with these commands:
```shell
cd CDG
mkdir data
# Convert data for validation if preparing for the first time
python tools/convert_datasets/gta.py data/gta # Source domain
python tools/convert_datasets/cityscapes.py data/cityscapes
# Convert Mapillary to Cityscapes format and resize for validation
python tools/convert_datasets/mapillary2cityscape.py data/mapillary data/mapillary/cityscapes_trainIdLabel --train_id
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images data/mapillary/cityscapes_trainIdLabel/val/label data/mapillary/half/val_img data/mapillary/half/val_label
```
(Optional) **ACDC**: Download all image and label packages from [ACDC](https://acdc.vision.ee.ethz.ch/) and extract them to `data/acdc`.

The final folder structure should look like this:

```
CDG
├── ...
├── checkpoints
│   ├── dinov2_vitl14_pretrain.pth
│   ├── dinov2_rein_and_head.pth
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── bdd100k
│   │   ├── images
│   │   |   ├── 10k
│   │   │   |    ├── train
│   │   │   |    ├── val
│   │   ├── labels
│   │   |   ├── sem_seg
│   │   |   |    ├── masks
│   │   │   |    |    ├── train
│   │   │   |    |    ├── val
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscapes_trainIdLabel
│   │   ├── half
│   │   │   ├── val_img
│   │   │   ├── val_label
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ...
```
## Pretraining Weights
* **Download:** Download pre-trained weights from [Rein](https://github.com/w1oves/Rein) for testing. Place them in the project directory without changing the file name.
* **Convert:** Convert pre-trained weights for training or evaluation.
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted.pth
  ```
  (optional for 1024x1024 resolution)
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted_1024x1024.pth --height 1024 --width 1024
  ```
## Evaluation (refer to Rein)
  Run the evaluation:
  ```
  python tools/test.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py checkpoints/dinov2_rein_and_head.pth --backbone dinov2_converted.pth
  ```
  For most of provided release checkpoints, you can run this command to evluate
  ```
  python tools/test.py /path/to/cfg /path/to/checkpoint --backbone /path/to/dinov2_converted.pth #(or dinov2_converted_1024x1024.pth)
  ```

## Training (refer to Rein)
Start training in single GPU:
```
python tools/train.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py
```
Start training in multiple GPU:
```
PORT=12345 CUDA_VISIBLE_DEVICES=1,2,3,4 bash tools/dist_train.sh configs/dinov2/rein_dinov2_mask2former_1024x1024_bs4x2.py NUM_GPUS
```

## Generate full weights (refer to Rein)
Because we only fine-tune and save the REIN and head weights, if you need a complete set of segmentor weights, you need to use this script:
```
python generate_full_weights.py --segmentor_save_path SEGMENTOR_SAVE_PATH --backbone CONVERTED_BACKBONE --rein_head REIN_HEAD
```

## Acknowledgment
Our implementation is mainly based on following repositories. Thanks for their authors.
* [Rein](https://github.com/w1oves/Rein)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
