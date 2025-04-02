# StyleGAN2-ADA custom implementation for medical data

./docker_run.sh python3 train_v2.py --outdir=training-runs/ --data=out/hyperkvasir --gpus=1 --cfg=stylegan2_masks_v1 --aug=noaug --metrics=fid5k_full --kimg=2500 --cond=0 --batch=32

## Requirements

1. Our dataset is required to be modified with the stylegan2_datamaker. This ensures the correct folder organization and format in order for the dataset_tool_v2 to work properly.

## Getting started

1. Use the stylegan2_datamaker to modify your dataset folder and create the necessary files.
2. Copy the dataset folder inside the "datasets" folder.
3. Change the source argument when running dataset_tool_v2 to use the dataset of choice.

## Changes in dataset_tool_v2

**open_image_folder**

- Addition of 3rd argument (dataset_name).

**convert_dataset**

- A txt file is created in the out folder called "info.csv", containing the current dataset name.

## Changes in train_v2
