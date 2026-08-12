import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

# ==================================================
# 1. НАСТРОЙКА АУГМЕНТАЦИИ
# ==================================================
# datagen = ImageDataGenerator(
#     rotation_range=15,  # Поворот
#     width_shift_range=0.15,  # Сдвиг по X
#     height_shift_range=0.15,  # Сдвиг по Y
#     zoom_range=0.15,  # Масштабирование
#     shear_range=5,  # Наклон (полезно для рукописного текста)
#     validation_split=0.2,  # 20% на валидацию
#     fill_mode='constant',  # Заполнение пустот белым
#     cval=255  # Значение для fill_mode='constant' (белый)
#)

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.07,
    height_shift_range=0.07,
    zoom_range=0.05,
    #shear_range=3,
    validation_split=0.2,
    # fill_mode='constant',
    # cval=255
)

# ==================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==================================================
data_dir = pathlib.Path(__file__).parent / "dataset"

# Загружаем маппинг классов
with open(data_dir / "class_mapping.json", 'r') as f:
    class_mapping = json.load(f)

num_classes = len(class_mapping)
print(f"Найдено классов: {num_classes}")
print(f"Классы: {list(class_mapping.values())}")

# Загрузка обучающей выборки
train_ds = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),  # 64×64 для лучшего распознавания
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',  # Многоклассовая
    subset='training',
    shuffle=True
)

# Загрузка валидационной выборки
val_ds = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',  # Многоклассовая
    subset='validation',
    shuffle=False
)

print(f"✅ Обучающих: {train_ds.samples}, Валидационных: {val_ds.samples}")

# ==================================================
# 3. СОЗДАНИЕ НЕЙРОСЕТИ (улучшенная архитектура)
# ==================================================
model = keras.Sequential([
    # Входной слой
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Второй блок
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Третий блок (для сложных символов)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Четвертый блок
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    # Полносвязная часть
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Выходной слой (21 класс)
    layers.Dense(num_classes, activation='softmax')  #softmax для многоклассовой
])

# Компиляция
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  #многоклассовая
    metrics=['accuracy']
)

model.summary()

# ==================================================
# 4. КОЛЛБЭКИ
# ==================================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12,
        min_lr=0.00001,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
]

# ==================================================
# 5. ОБУЧЕНИЕ
# ==================================================
print("Начало обучения...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# ==================================================
# 6. ГРАФИКИ ОБУЧЕНИЯ
# ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Точность на обучении')
ax1.plot(history.history['val_accuracy'], label='Точность на проверке')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Точность')
ax1.set_title('Точность модели')
ax1.legend()

ax2.plot(history.history['loss'], label='Потери на обучении')
ax2.plot(history.history['val_loss'], label='Потери на проверке')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Потери')
ax2.set_title('Функция потерь')
ax2.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# ==================================================
# 7. КОНВЕРТАЦИЯ В TENSORFLOW LITE
# ==================================================
print("Конвертирую в TFLite...")


# Загружаем лучшую модель
model.load_weights('best_model.h5')

# Создаём конвертер
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Только базовая оптимизация (без потери точности)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Явно указываем, что вход 64x64 и тип float32
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# Конвертируем
tflite_model = converter.convert()

# Сохраняем
with open('calculator_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Модель сохранена как calculator_model.tflite")
print(f"Размер: {len(tflite_model) / 1024:.2f} KB")

# Проверяем размер входа
import tensorflow.lite as tflite
interpreter = tflite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f" Вход TFLite: {input_details[0]['shape']}, тип: {input_details[0]['dtype']}")
print(f" Выход TFLite: {output_details[0]['shape']}, тип: {output_details[0]['dtype']}")

# # Загружаем лучшую модель
# model.load_weights('best_model.h5')
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
# # Оптимизации для уменьшения размера
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
#
# # # Квантование (уменьшаем размер модели)
# # def representative_dataset():
# #     for _ in range(100):
# #         yield [tf.random.normal([1, 64, 64, 1])]
#
#
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
#
# tflite_model = converter.convert()
#
#
# # Проверяем размер входа модели
# import tensorflow.lite as tflite
# interpreter = tflite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# print(f"Размер входа TFLite: {input_details[0]['shape']}")
#
# # Сохраняем
# with open('calculator_model.tflite', 'wb') as f:
#     f.write(tflite_model)
#
# print("Модель сохранена как calculator_model.tflite")
# print(f"Размер: {len(tflite_model) / 1024:.2f} KB")


# # ====== КОНВЕРТАЦИЯ В TFLite ======
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
# # Явно указываем размер входа
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
#
# # Оптимизация
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
# # Указываем точный размер входного тензора
# def representative_dataset():
#     for _ in range(100):
#         yield [tf.random.normal([1, 64, 64, 1])]
#
# converter.representative_dataset = representative_dataset
#
# tflite_model = converter.convert()
#
# with open('calculator_model.tflite', 'wb') as f:
#     f.write(tflite_model)
#
# print("Модель сохранена как calculator_model.tflite")





# ==================================================
# 8. СОХРАНЯЕМ МАППИНГ КЛАССОВ ДЛЯ ANDROID
# ==================================================
# Инвертируем маппинг для Android
# android_mapping = {v: int(k) for k, v in class_mapping.items()}
# with open('class_mapping_android.json', 'w') as f:
#     json.dump(android_mapping, f, indent=2)
#
# print("Маппинг классов сохранен как class_mapping_android.json")



