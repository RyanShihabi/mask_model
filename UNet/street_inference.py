import segmentation_models_pytorch as smp
import cv2
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load("./street_unet.pth", weights_only=True))
model.eval()

img = cv2.imread("./satellite_house_cropped/33.10892_-117.2476_1560000.png")
img_resize = cv2.resize(img, (512, 512)) / 255.
img_t = torch.Tensor(img_resize).permute((2, 0, 1)).unsqueeze(dim=0).to(device)

print(img_t.shape)

with torch.no_grad():
    output = model(img_t).sigmoid()

    mask = (output > 0.5).float().squeeze(dim=0).permute((1,2,0)).cpu().numpy()
    mask = cv2.resize(mask, img.shape[:2][::-1]).astype(np.uint8)

    colored_mask = np.zeros_like(img)
    colored_mask[:, :] = [0, 255, 0]

    colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)

    alpha = 0.25
    masked_image = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)

    cv2.imshow("mask", masked_image)
    cv2.waitKey(0)