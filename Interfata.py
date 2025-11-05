import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import pydicom
import numpy as np

class FriendlyCancerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Health Assistant - Detectare Cancer Mamar")
        self.geometry("650x550")
        ctk.set_appearance_mode("light")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "vit-modelv2"  # Înlocuiește cu calea corectă
        self.model = ViTForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)

        self.bg_frame = ctk.CTkFrame(self, fg_color="#ffe6f0", corner_radius=0)
        self.bg_frame.pack(fill="both", expand=True)

        # Frame central
        self.center_frame = ctk.CTkFrame(self.bg_frame, fg_color="#ffd9ec", corner_radius=15, width=750, height=800)
        self.center_frame.place(relx=0.5, rely=0.8, anchor="center")

        self.label_title = ctk.CTkLabel(
            self.center_frame,
            text="🌸 AI Health Assistant",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="#4a235a"
        )
        self.label_title.pack(pady=(10, 5))

        self.subtitle = ctk.CTkLabel(
            self.center_frame,
            text="Încarcă o imagine și detectăm automat semnele de cancer mamar",
            font=ctk.CTkFont(size=16),
            text_color="#6b3a6b",
            wraplength=480,
            justify="center"
        )
        self.subtitle.pack(pady=(0, 15), padx=20)

        self.button_upload = ctk.CTkButton(self.center_frame, text="📁 Încarcă Imagine", command=self.upload_image, width=220)
        self.button_upload.pack(pady=(0, 10))

        self.decorative_frame = ctk.CTkFrame(self.bg_frame, fg_color="transparent")
        self.decorative_frame.place(relx=0.5, rely=0.3, anchor="center")

        # Frame pentru imagine + rezultat
        self.image_frame = ctk.CTkFrame(self.bg_frame, fg_color="#ffd9ec", corner_radius=15, width=300, height=500)
        self.image_frame.place_forget()

        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(pady=10, padx=40)

        self.result_label = ctk.CTkLabel(self.image_frame, text="", font=ctk.CTkFont(size=18, weight="bold"))
        self.result_label.pack(pady=10, padx=40)

        # Imaginea doctorului
        try:
            doctor_img = Image.open("doctor.png").resize((400, 400))
        except:
            doctor_img = Image.new("RGBA", (400, 400), (255, 192, 203, 0))

        self.doctor_photo = ImageTk.PhotoImage(doctor_img)
        self.label_doctor = ctk.CTkLabel(self.bg_frame, image=self.doctor_photo, text="")
        self.label_doctor.place(relx=0.5, rely=0.35, anchor="center")

        self.button_analyze = None
        self.button_back = ctk.CTkButton(
            self.bg_frame,
            text="⬅️ Înapoi",
            command=self.go_back,
            width=120,
            fg_color="#ff85b3",
            hover_color="#ff5599",
            text_color="white"
        )
        self.button_back.place_forget()

        self.image_path = None
        self.photo_image = None

    def load_image(self, path):
        if path.lower().endswith('.dcm'):
            ds = pydicom.dcmread(path)
            img_array = ds.pixel_array
            # Normalizare în interval 0-255
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255
            img_array = img_array.astype(np.uint8)
            image = Image.fromarray(img_array).convert("RGB")
            return image
        else:
            return Image.open(path).convert("RGB")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imagini", "*.jpg *.jpeg *.png *.bmp *.dcm")])
        if file_path:
            self.image_path = file_path
            self.image_label.bind("<Button-1>", lambda e: self.change_image())

            # Ascunde elementele UI
            self.center_frame.place_forget()
            self.decorative_frame.place_forget()
            self.label_doctor.place_forget()

            # Afișează frame-ul imaginii
            self.image_frame.place(relx=0.5, rely=0.4, anchor="center")
            self.button_back.place(relx=0.5, rely=0.95, anchor="center")

            pil_image = self.load_image(file_path)
            pil_image = ImageOps.contain(pil_image, (400, 400))
            background = Image.new('RGB', (400, 400), (255, 217, 236))
            offset = ((400 - pil_image.width) // 2, (400 - pil_image.height) // 2)
            background.paste(pil_image, offset)

            self.photo_image = ImageTk.PhotoImage(background)
            self.image_label.configure(image=self.photo_image, text="")

            # Buton analiză
            if self.button_analyze is None:
                self.button_analyze = ctk.CTkButton(
                    self.bg_frame,
                    text="🔍 Analizează Imaginea",
                    command=self.predict,
                    width=250
                )
            self.button_analyze.place(relx=0.5, rely=0.85, anchor="center")
            self.result_label.configure(text="")

    def change_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imagini", "*.jpg *.jpeg *.png *.bmp *.dcm")])
        if file_path:
            self.image_path = file_path
            pil_image = self.load_image(file_path)
            pil_image = ImageOps.contain(pil_image, (400, 400))
            background = Image.new('RGB', (400, 400), (255, 217, 236))
            offset = ((400 - pil_image.width) // 2, (400 - pil_image.height) // 2)
            background.paste(pil_image, offset)

            self.photo_image = ImageTk.PhotoImage(background)
            self.image_label.configure(image=self.photo_image, text="")
            self.result_label.configure(text="")

    def go_back(self):
        self.image_frame.place_forget()
        if self.button_analyze:
            self.button_analyze.place_forget()
        self.button_back.place_forget()
        self.result_label.configure(text="")
        self.center_frame.place(relx=0.5, rely=0.8, anchor="center")
        self.decorative_frame.place(relx=0.5, rely=0.3, anchor="center")
        self.label_doctor.place(relx=0.5, rely=0.35, anchor="center")

    def predict(self):
        if self.image_path:
            image = self.load_image(self.image_path)
            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

            label = self.model.config.id2label[predicted_class_idx].lower()

            if label == "normal":
                text = "✔️ Imagine normală – fără semne de cancer"
                color = "green"
            elif label == "benign" or label == "benign":
                text = "⚠️ Posibil benign – nu este cancer malign"
                color = "orange"
            elif label == "malignant":
                text = "⚠️ Posibil cancer malignant detectat"
                color = "red"
            else:
                text = f"⚠️ Clasificare necunoscută: {label}"
                color = "#4a235a"

            self.result_label.configure(text=text, text_color=color)

if __name__ == "__main__":
    app = FriendlyCancerApp()
    app.mainloop()
