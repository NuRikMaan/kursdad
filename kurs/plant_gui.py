import os
import shutil
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# ---------------- Настройки ----------------
UPLOAD_DIR = "uploads"
IMG_SIZE = (256, 256)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Модели ----------------
DISEASE_MODEL_PATH = "models/tomatoes.h5"
disease_model = load_model(DISEASE_MODEL_PATH, compile=False)
DISEASE_CLASSES = [
    "Бактериальный ожог",
    "Ранняя фитофтороз",
    "Поздняя фитофтороз",
    "Септориоз",
    "Мучнистая роса",
    "Вершинная гниль",
    "Здоровое растение"
]

INSECT_MODEL_PATH = "models/best.pt"
insect_model = YOLO(INSECT_MODEL_PATH)
INSECT_CLASSES = ["Паутинный клещ", "Ногохвостка", "Тля"]

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Диагностика болезней томатов и вредителей")
root.geometry("500x600")
root.configure(bg="#f0f0f5")

selected_file_path = None

# ---------------- Функции ----------------
def select_image():
    global selected_file_path
    file_path = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    selected_file_path = file_path
    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk
    result_text.delete('1.0', tk.END)

def save_and_predict():
    global selected_file_path
    if not selected_file_path:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, "Сначала выберите фото!\n")
        return
    new_name = name_entry.get().strip()
    if not new_name:
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, "Введите имя для сохранения!\n")
        return

    # Сохраняем фото
    ext = os.path.splitext(selected_file_path)[1]
    save_path = os.path.join(UPLOAD_DIR, f"{new_name}{ext}")
    shutil.copy(selected_file_path, save_path)

    # Предсказание болезни
    img = Image.open(selected_file_path).resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    disease_preds = disease_model.predict(arr)
    disease_idx = np.argmax(disease_preds)
    disease_conf = disease_preds[0][disease_idx] * 100
    disease_class = DISEASE_CLASSES[disease_idx]
    if disease_class == "Здоровое растение":
        disease_result = f"{disease_class} ({disease_conf:.1f}%)"
    else:
        disease_result = f"Болен: {disease_class} ({disease_conf:.1f}%)"

    # Предсказание насекомых
    results = insect_model.predict(selected_file_path)[0]
    insect_predictions = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            if cls_idx < len(INSECT_CLASSES):
                insect_predictions.append(f"{INSECT_CLASSES[cls_idx]} ({conf:.1f}%)")
            else:
                insect_predictions.append(f"Неизвестный ({conf:.1f}%)")
    else:
        insect_predictions.append("Вредитель не обнаружен")

    # Вывод результатов
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, f"{disease_result}\nВредители:\n" + "\n".join(insect_predictions))

# ---------------- Виджеты ----------------
title_label = tk.Label(root, text="Диагностика томатов", font=("Arial", 20, "bold"), bg="#f0f0f5", fg="#333")
title_label.pack(pady=10)

panel = tk.Label(root, bg="#d9d9d9", relief="groove", bd=2)
panel.pack(pady=10)

btn_select = tk.Button(root, text="Выбрать фото", font=("Arial", 12), bg="#4CAF50", fg="white", command=select_image)
btn_select.pack(pady=5, ipadx=10, ipady=5)

name_entry = tk.Entry(root, font=("Arial", 12), width=30, bd=2, relief="groove")
name_entry.pack(pady=5)
name_entry.insert(0, "Введите имя файла здесь")

btn_save = tk.Button(root, text="Сохранить и предсказать", font=("Arial", 12), bg="#2196F3", fg="white", command=save_and_predict)
btn_save.pack(pady=10, ipadx=10, ipady=5)

result_text = scrolledtext.ScrolledText(root, font=("Arial", 12), width=50, height=10, bd=2, relief="sunken")
result_text.pack(pady=10)

root.mainloop()