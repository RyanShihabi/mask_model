from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import joblib
import base64

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Segmentation Model
segmentation_model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

segmentation_model.load_state_dict(torch.load("./street_unet.pth", weights_only=True))

xgboost_mask_model = joblib.load("./xgboost_mask.joblib")
xgboost_mask_model.device = device

# Is Alive?
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Model server is running"}), 200

# Satellite Image Segmentation and Value Prediction
@app.route("/satellite_predict", methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image'].read()

    try:
        # Decode image from POST call
        image_np = np.frombuffer(image_file, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        img_resize = cv2.resize(image, (512, 512)) / 255.

        img_t = torch.Tensor(img_resize).permute((2, 0, 1)).unsqueeze(dim=0).to(device)

        # Inference
        segmentation_model.eval()
        with torch.no_grad():
            output = segmentation_model(img_t).sigmoid()

        # Convert to boolean mask
        sigmoid_mask = (output > 0.5).float()

        # Make mask opencv readable
        mask = sigmoid_mask.squeeze(dim=0).permute((1,2,0)).cpu().numpy()
        mask = cv2.resize(mask, image.shape[:2][::-1]).astype(np.uint8)

        # Calculate mask proportion
        street_coef = mask.sum() / (mask.shape[0] * mask.shape[1])

        # Visualize mask
        colored_mask = np.zeros_like(image)
        colored_mask[:, :] = [0, 255, 0]

        colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)

        alpha = 1
        masked_image = cv2.addWeighted(colored_mask, 0.4, image, alpha, 0)

        # Encode image for request
        _, img_b = cv2.imencode(".png", masked_image)
        img_b = img_b.tobytes()
        img_b64 = base64.b64encode(img_b).decode()
        
        # Feed mask to xgboost model
        xgboost_input = sigmoid_mask[0][0].cpu().numpy().astype(np.uint8)

        mask_resize = cv2.resize(xgboost_input, (64, 64))

        xgboost_input = [mask_resize.flatten().tolist()]
        
        pred = xgboost_mask_model.predict(xgboost_input).tolist()[0]

        # Return payload
        response = {"segmentation_image": img_b64, "prediction": f"${pred:,.2f}", "coef": street_coef}

    except Exception as e:
        print(e)
        response = {"error": str(e)}

    return jsonify(response), 200

# Run on public IP from port 4000
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000, debug=True)
