import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

model_0 = torch.load("./model_0.pth")
model_0.eval() 

def classify_image(image):
    """
    Classifies an input image using a PyTorch model.

    Parameters:
    image (PIL.Image.Image): The input image to be classified.

    Returns:
    int: The predicted class index based on the model's classification.
    """

    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = (image_tensor)

    _, predicted_idx = torch.max(outputs, 1)
    predicted_class = predicted_idx.item()
    
    return predicted_class

