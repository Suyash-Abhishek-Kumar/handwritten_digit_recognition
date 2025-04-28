import torch #type:ignore 
import gradio as gr #type:ignore
import torch.nn as nn #type:ignore
import numpy as np #type:ignore
from PIL import Image, ImageOps #type:ignore
from torchvision import transforms #type:ignore

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(24*24*64, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()


def mnist_style_preprocess(img):
    img = 255 - img.astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L')
    # Binarize
    pil_img = pil_img.point(lambda p: 255 if p > 50 else 0)
    # Find bounding box
    bbox = pil_img.getbbox()
    if bbox:
        pil_img = pil_img.crop(bbox)
    # Resize to 20x20, keeping aspect ratio
    pil_img = ImageOps.contain(pil_img, (20, 20), Image.Resampling.LANCZOS)
    # Paste into center of 28x28
    new_img = Image.new('L', (28, 28), 0)
    upper_left = ((28 - pil_img.width) // 2, (28 - pil_img.height) // 2)
    new_img.paste(pil_img, upper_left)
    return new_img
def predict_digit(img):
    try:
        # Handle dict input (Gradio EditorValue)
        if isinstance(img, dict):
            if 'composite' in img and img['composite'] is not None:
                img = img['composite']
            elif 'background' in img and img['background'] is not None:
                img = img['background']
            else:
                return "No image found in dict input."
        if hasattr(img, 'ndim') and img.ndim == 3:
            img = img[..., 0]
        processed_img = mnist_style_preprocess(img)
        img_tensor = transform(processed_img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        return str(output.argmax(dim=1).item())
    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a digit (0-9)", type="numpy"),
    outputs="label",
    live=True,
    title="MNIST Digit Classifier",
    description="Draw a digit between 0-9 in the box below"
)

demo.launch()
