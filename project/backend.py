from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
from torchvision import transforms
from models.dvt_model import create_dvt_model
import os

app = FastAPI()

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
model = create_dvt_model(num_classes=num_classes).to(device)

model_path = './checkpoints/best_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
else:
    print("Warning: Model weights not found. Running with random initialization.")

model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    return {
        "class": CLASSES[predicted.item()],
        "confidence": float(confidence.item()) * 100
    }

@app.post("/reload")
async def reload_model():
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        return {"status": "Model reloaded successfully"}
    return {"status": "Error: File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
