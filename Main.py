from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Încarcă modelul și procesorul
model = ViTForImageClassification.from_pretrained("vit-model")
processor = ViTImageProcessor.from_pretrained("vit-model")

# Încarcă o imagine
image = Image.open("data/val/normal/1-1_19.jpg").convert("RGB")

# Preprocesează imaginea
inputs = processor(images=image, return_tensors="pt")

# Predicție
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Eticheta clasei
label = model.config.id2label[predicted_class_idx]
print("Predicțiaeste:", label)
