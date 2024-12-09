# Satellite Image Street Mask Model

Scraping satellite images for a street segmentation model

Model included:
1. UNet
2. XGBoost

Dataset
1. `python3 scripts/cvat2mask.py --zip streets1122.zip --image_dir satellite_house_cropped`

Training:
1. UNet/UNet_train
2. XGBoost/Street_XGBoost.ipynb

Inference:
1. UNet/street_inference.py
2. api.py

API
`python3 api.py`

App
`shiny run app.py`

Model paths and datasets are included!

For the model weights and data too large for Github, please contact author