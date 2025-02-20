import tensorflow as tf
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import os

IMG_SIZE = 128

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def main():
    try:
        model_path = 'best_model.keras'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at {model_path}")
            return
        
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
        root = Tk()
        root.withdraw()
        
        while True:
            file_path = filedialog.askopenfilename(
                title="Select an image to classify",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                break
            
            img = load_and_preprocess_image(file_path)
            prediction = model.predict(img, verbose=0)[0][0]
            gender = "Female" if prediction > 0.5 else "Male"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            print(f"\n🖼️ Image: {os.path.basename(file_path)}")
            print(f"📊 Prediction: {gender} (confidence: {confidence:.2%})")
            
            response = input("\nWould you like to classify another image? (y/n): ")
            if response.lower() != 'y':
                break
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()