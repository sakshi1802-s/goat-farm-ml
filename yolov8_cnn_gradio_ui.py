import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import traceback

print("Starting system initialization")

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# MODEL
class GoatCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GoatCNN, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_head(x)
        return x

# LOAD YOLO
print("Loading YOLOv8...")
try:
    yolo_model = YOLO("yolov8_goat_detector.pt")
    print("✓ YOLOv8 loaded!")
except:
    print("✗ YOLOv8 not found")
    yolo_model = None

# LOAD CNN
print("Loading CNN...")
try:
    cnn_model = GoatCNN(num_classes=10).to(device)
    cnn_model.eval()
    state_dict = torch.load("goat_recognition_cnn.pth", map_location=device)
    cnn_model.load_state_dict(state_dict)
    print("✓ CNN loaded!")
except Exception as e:
    print(f"✗ CNN error: {e}")
    cnn_model = None

goat_ids = ['10','47','48','64','68','94','95','108','304','688']


def recognize_goat(image_input):
    try:
        if image_input is None:
            return None, "No image provided"

        # convert
        image_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        results = "GOAT RECOGNITION RESULTS\n"
        results += "="*40 + "\n\n"

        # YOLO
        results += "[Detection]\n"
        detections = yolo_model.predict(source=image_cv, conf=0.5, verbose=False)
        detection = detections[0]

        
        for box in detection.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        results += f"Detected: {len(detection.boxes)} object(s)\n\n"

        
        results += "[Classification]\n"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cnn_model(image_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        goat_id = goat_ids[pred_idx.item()]
        conf_score = conf.item()

        results += f"Goat ID: {goat_id}\n"
        results += f"Confidence: {conf_score*100:.2f}%\n"

        return image_pil, results

    except Exception as e:
        return None, str(e)

# UI
interface = gr.Interface(
    fn=recognize_goat,
    inputs=gr.Image(type="pil", label="Upload Goat Image"),
    outputs=[
        gr.Image(label="Detection Output"),
        gr.Textbox(label="Results")
    ],
    title="Goat Biometric Recognition System",
    description="YOLOv8 + CNN based goat identification"
)

# RUN
if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)