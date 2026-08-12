import tensorflow as tf
import numpy as np
import cv2
import os
import json

# ====== 1. ЗАГРУЖАЕМ МОДЕЛЬ ======
model = tf.keras.models.load_model('best_model.h5') #.models. = Файл в папке models внутри текущей
# .load_model = Открывает файл и читает его содержимое. Восстанавливает архитектуру модели (слои, связи, функции активации).
print("✅ Модель загружена")

# ====== 2. ЗАГРУЖАЕМ МАППИНГ ======
with open('class_mapping_android.json', 'r') as f:
    mapping = json.load(f) # json.load() = для чтения данных из файла и их преобразования в словарь или список Python.
idx_to_name = {int(k): v for k, v in mapping.items()} # .items() возвращает пары (ключ, значение) в виде кортежей
print(f"📋 Классы: {idx_to_name}")

# ====== 3. ЗАГРУЖАЕМ КАРТИНКИ ИЗ ANDROID ======
test_dir = 'android_test'
images = []
filenames = []

for img_file in os.listdir(test_dir):
    if not img_file.endswith('.png'): # .endswith('.png') заканчивается ли строка на пнг
        continue
    img_path = os.path.join(test_dir, img_file) # склеивает путь к папке и имя файла в один корректный путь. .path = выбирает правильный слеш (/ или \)

    # Загружаем картинку (именно так, как её отдаёт Android)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # cv2.imread() — функция из библиотеки OpenCV, которая загружает картинку с диска.
# cv2.IMREAD_GRAYSCALE — говорит: «Загрузи картинку как чёрно-белую (в оттенках серого)».
# Что получаем: Двумерный массив (матрица) размером (высота, ширина), где каждый пиксель — число от 0 до 255 (0 — чёрный, 255 — белый).

    # ПРОВЕРЯЕМ: если картинка уже 64x64 — не ресайзим!
    if img.shape != (64, 64):
        img = cv2.resize(img, (64, 64)) # изменяет размер картинки

    # Нормализация (как в Android)
    img = img.astype(np.float32)

    images.append(img)
    filenames.append(img_file)

if not images:
    print("❌ Нет картинок в папке android_test")
    exit()

X = np.array(images).reshape(-1, 64, 64, 1) # -1 Автоматически подставить количество картинок
# ====== 4. ПРЕДСКАЗАНИЕ ======
print(f"\n🔍 Проверяю {len(X)} картинок из Android...")
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)# .argmax вернуть максимальное значение индекса из строки предсказания
confidences = np.max(predictions, axis=1)# .argmax вернуть максимальное значение из строки предсказания

# ====== 5. ВЫВОД РЕЗУЛЬТАТОВ ======
print("\n📊 Результаты:")
for i, (filename, pred_idx, confidence) in enumerate(zip(filenames, predicted_classes, confidences)): # Функция zip работает как механическая застежка-молния. Она берет первые элементы из всех трех списков и склеивает их в одну группу (кортеж), затем вторые, третьи и так далее.
    # Функция enumerate добавляет к каждой склеенной группе автоматический счетчик (индекс), начиная с нуля.
    pred_name = idx_to_name.get(pred_idx, '?')# ищет имя класса по индексу, если не находит — возвращает ?.
    print(f"  {i + 1}: {filename} → {pred_name} (уверенность: {confidence:.2f})") #   1: 1781841703771.png → 2 (уверенность: 0.90)

# ====== 6. ВЫВОД СТАТИСТИКИ ======
print("\n📊 Статистика:")
unique, counts = np.unique(predicted_classes, return_counts=True) # возвращает не только уникальные значения, но и их количество.
for idx, count in zip(unique, counts):
    name = idx_to_name.get(idx, '?')
    print(f"  {name}: {count} раз")

