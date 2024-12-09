import argparse
import shutil
import os
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import json
import pandas as pd

def rle2mask(rle: list, left: int, top: int, width: int, height: int, img_width: int, img_height: int, label: str):
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    decoded = np.zeros(width * height, dtype=np.uint8)

    current_position = 1
    value = 0

    for segment_length in rle:
        decoded[current_position:current_position + segment_length-1] = value
        current_position += segment_length
        value = 1 - value 

    decoded = np.array(decoded, dtype=np.uint8)
    decoded = decoded.reshape((height, width))

    mask[top:top+height, left:left+width] = decoded

    return mask

parser = argparse.ArgumentParser()

parser.add_argument("--zip", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)

args = parser.parse_args()

dir = args.zip.split(".")[0]

if os.path.exists(f"./{dir}"):
    shutil.rmtree(f"./{dir}")

shutil.unpack_archive(args.zip, f"./{dir}")

tree = ET.parse(f"./{dir}/annotations.xml")

root = tree.getroot()

dataset = {"data": []}

xgboost_dataset = []
xgboost_train_dataset = []
xgboost_val_dataset = []

for image in root.findall('image'):
    img_width = int(image.attrib["width"])
    img_height = int(image.attrib["height"])

    img_name = image.attrib["name"]

    canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    total_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    masks = image.findall("mask")

    if len(masks) == 0:
        continue
    
    for mask in masks:
        mask_attr = mask.attrib
        
        label = mask_attr["label"]

        rle = [int(count) for count in mask_attr["rle"].split(", ")]
        left = int(mask_attr["left"])
        top = int(mask_attr["top"])
        width = int(mask_attr["width"])
        height = int(mask_attr["height"])

        mask = rle2mask(rle, left, top, width, height, img_width, img_height, label)
        total_mask = total_mask | mask
    
    dataset["data"].append([os.path.join(args.image_dir, img_name), cv2.resize(src=total_mask, dsize=(512, 512)).tolist()])
    xgboost_dataset.append([*cv2.resize(total_mask, (256, 256)).flatten().tolist(), int(img_name.split("_")[-1].split(".")[0])])

label_split_index = np.random.default_rng().choice(len(dataset["data"]), size=len(dataset["data"]), replace=False).astype(int)

dataset_split = 0.8

train_dataset = [dataset["data"][i] for i in label_split_index[:int(len(dataset["data"])*dataset_split)]]
val_dataset = [dataset["data"][i] for i in label_split_index[int(len(dataset["data"])*dataset_split):]]

xgboost_train_dataset = [xgboost_dataset[i] for i in label_split_index[:int(len(xgboost_dataset)*dataset_split)]]
xgboost_val_dataset = [xgboost_dataset[i] for i in label_split_index[int(len(xgboost_dataset)*dataset_split):]]

with open("train_street_dataset.json", "w") as f:
    json.dump(train_dataset, f)
f.close()

with open("val_street_dataset.json", "w") as f:
    json.dump(val_dataset, f)
f.close()

train_df = pd.DataFrame(xgboost_train_dataset)
val_df = pd.DataFrame(xgboost_val_dataset)

train_df.to_csv("./train_street_dataset.csv")
val_df.to_csv("./val_street_dataset.csv")