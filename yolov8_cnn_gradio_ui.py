import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import traceback

print("Starting system initialization")


#  DEVICE DETECTION


if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "GPU (CUDA)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"> Device: {device_name}")


# LOAD TRAINED MODELS


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

# Load models
print("Loading YOLOv8...")
try:
    yolo_model = YOLO("yolov8_goat_detector.pt")
    print("✓ YOLOv8 loaded!")
except:
    print("✗ YOLOv8 not found")
    yolo_model = None

print("Loading CNN..")
try:
    cnn_model = GoatCNN(num_classes=10).to(device)
    cnn_model.eval()
    state_dict = torch.load("goat_recognition_cnn.pth", map_location=device)
    cnn_model.load_state_dict(state_dict)
    print("✓ CNN loaded!")
except Exception as e:
    print(f"✗ CNN error: {e}")
    cnn_model = None

goat_ids = ['10', '47', '48', '64', '68', '94', '95', '108', '304', '688']



# INFERENCE FUNCTION

def recognize_goat(image_input):
    """Recognize goat from image"""
    
    try:
        if image_input is None:
            return None, " No image provided"
        
        # Convert to OpenCV
        if isinstance(image_input, Image.Image):
            image_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_input
        
        results = " GOAT RECOGNITION RESULTS\n"
        results += "="*50 + "\n\n"
        
        # Detection
        results += "[STEP 1] YOLOv8 Face Detection\n"
        if yolo_model is None:
            results += " YOLOv8 model not loaded\n"
            return None, results
        
        try:
            detections = yolo_model.predict(source=image_cv, conf=0.5, verbose=False)
            detection = detections[0]
            num_faces = len(detection.boxes)
            results += f"✓ Detected {num_faces} face(s)\n\n"
        except Exception as e:
            results += f" Detection failed: {str(e)}\n"
            return None, results
        
        # Recognition
        results += "[STEP 2] CNN Individual Recognition\n"
        if cnn_model is None:
            results += " CNN model not loaded\n"
            return None, results
        
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
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
            
            results += f"✓ Predicted Goat ID: {goat_id}\n"
            results += f"✓ Confidence: {conf_score*100:.2f}%\n\n"
            
            # Top 3
            results += "[Top-3 Predictions]\n"
            top3_probs, top3_idx = torch.topk(probs[0], 3)
            for i, (p, idx) in enumerate(zip(top3_probs, top3_idx), 1):
                results += f"{i}. Goat {goat_ids[idx.item()]}: {p.item()*100:.2f}%\n"
            
            results += "\n[Confidence Level]\n"
            if conf_score >= 0.70:
                results += " HIGH - Accept prediction"
            elif conf_score >= 0.50:
                results += " MEDIUM - Manual review needed"
            else:
                results += " LOW - Multiple confirmations needed"
            
            return image_pil, results
            
        except Exception as e:
            results += f" Recognition failed: {str(e)}\n"
            traceback.print_exc()
            return None, results
    
    except Exception as e:
        return None, f" Error: {str(e)}\n{traceback.format_exc()}"


# GRADIO UI


print("Creating Gradio interface...")

interface = gr.Interface(
    fn=recognize_goat,
    inputs=gr.Image(label="Upload Goat Image", type="pil"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Results", lines=15)
    ],
    title=" Goat Biometric Recognition System",
    description="Upload a goat image to detect and identify individual goats using YOLOv8 + CNN",
    examples=None,
    theme="soft"
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAUNCHING GRADIO UI")
    print("="*60)
    print("\n📱 Open browser at: http://127.0.0.1:7860\n")
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=False,

    )
