from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import cv2
import json
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes

app = Flask(__name__)
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@app.cache.memoize(timeout=60 * 60 * 24)  # Cache for 24 hours
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval();
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    img = Image.open(file)
    prediction = make_prediction(np.array(img))
    img_with_bbox = draw_bounding_boxes(img, boxes=prediction["boxes"], labels=prediction["labels"],
                                        colors=["red" if label=="person" else "green" for label in prediction["labels"]], width=2)
    img_with_bbox = ToTensor()(img_with_bbox).permute(1, 2, 0)
    img_with_bbox = (img_with_bbox * 255).byte().numpy()
    img_with_bbox = cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR)

    _, img_encoded = cv2.imencode('.png', img_with_bbox)
    return jsonify({'image': img_encoded.tobytes()})

if __name__ == '__main__':
    app.run(debug=True)