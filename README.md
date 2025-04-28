# ✍️ Handwritten Digit Classifier (Real-Time Sketch Recognition)

A real-time handwritten digit recognition app built with Python, Pygame, and a CNN model!  
Draw a digit (0-9) in the sketchbox and let the trained neural network predict what you wrote — instantly!

Perfect for learning about Convolutional Neural Networks (CNNs) and real-time ML deployment!

---

## ✨ Features

- **Real-Time Prediction**: Draw digits and get instant feedback from the model.
- **Interactive Sketchpad**: Simple and intuitive interface to draw with your mouse.
- **Custom Trained Model**: CNN trained on a handwritten digits dataset for accurate predictions.
- **Seamless Integration**: Model training and GUI are neatly separated into different files.

---

## 📸 Screenshots

<img src="https://github.com/user-attachments/assets/ef4baa52-bf3d-4ee9-af6b-548ede87d7ff" alt="screenshot1" width="400"/>
<img src="https://github.com/user-attachments/assets/d88066b8-ea37-4585-a8c0-b3ddf98fcd4f" alt="screenshot2" width="400"/>
<img src="https://github.com/user-attachments/assets/5f37e9fb-d7d6-4df9-bbb9-aa760db2ce40" alt="screenshot3" width="400"/>
<img src="https://github.com/user-attachments/assets/450a3a9f-a73d-4539-95b5-914f1c25f3c5" alt="screenshot4" width="400"/>



---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/handwritten-digit-classifier.git
cd handwritten-digit-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

First, ensure you have the trained model (`model.pth`) ready.  
Then, simply run:

```bash
python digit_recog.py
```

Start drawing in the sketchbox and watch the model guess your digit!

---

## 🎮 Controls

- **Draw**: Hold left-click and move the mouse inside the sketchbox to draw.
- **Predict**: The prediction updates automatically as you draw.
- **Clear**: Press the "Clear" button to reset the sketchbox.

---

## 📁 Project Structure

```
handwritten-digit-classifier/
├── Model.ipynb       # Jupyter Notebook: CNN model creation and training
├── model.pth         # Trained CNN model (saved weights)
├── digit_recog.py    # GUI and real-time prediction logic
├── data/             # Handwritten digits dataset
├── requirements.txt  # List of required Python modules
```

---

## 📦 Dependencies

- Python 3.x  
- `pygame`  
- `torch` (PyTorch)  
- `torchvision`  
- (Other modules listed in `requirements.txt`)

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧠 How It Works

- **Model Training**: `Model.ipynb` defines and trains a Convolutional Neural Network (CNN) on the digit dataset.
- **Model Saving**: The trained model is saved as `model.pth`.
- **Prediction**: `digit_recog.py` loads `model.pth`, provides a sketchpad, and predicts the drawn digit in real-time.

---

## 🎨 Customization

- **Model**: Retrain `Model.ipynb` with a different dataset or tweak CNN layers for experimentation.
- **Sketchpad UI**: Modify `digit_recog.py` to change canvas size, brush thickness, colors, etc.

---

## 🙏 Acknowledgements

Created and maintained by **Your Name Here**.

Draw, predict, and explore the fascinating world of CNNs and real-time machine learning! ✨
