BY
ahmedsaleh243

Original file is located at
https://colab.research.google.com/drive/17xCjRGmuUn2uOftw4wSBt2BKLjF_lERs?usp=sharing


import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


hf_dataset_path = "Francesco/bone-fracture-7fylg"
print(f"جاري تحميل البيانات من {hf_dataset_path}...")
dataset = load_dataset(hf_dataset_path)

train_data = dataset['train']
val_data = dataset['validation']

def preprocess_image(element):
    image = element['image']
    objects = element['objects']

    label = 1 if len(objects['category']) > 0 else 0

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = tf.image.resize(tf.keras.preprocessing.image.img_to_array(image), [224, 224])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

def gen_train():
    for ex in train_data: yield preprocess_image(ex)

def gen_val():
    for ex in val_data: yield preprocess_image(ex)

output_sig = (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
              tf.TensorSpec(shape=(), dtype=tf.int64))

batch_size = 32
train_ds = tf.data.Dataset.from_generator(gen_train, output_signature=output_sig).batch(batch_size).repeat()
val_ds = tf.data.Dataset.from_generator(gen_val, output_signature=output_sig).batch(batch_size).repeat()

steps_per_epoch = max(1, len(train_data) // batch_size)
validation_steps = max(1, len(val_data) // batch_size)


# 2. (CNN Architecture & Transfer Learning)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

data_augmentation = Sequential([
  RandomFlip("horizontal_and_vertical"),
  RandomRotation(0.2),
], name="Data_Augmentation_Layer")

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D(name="Average_Pooling")(x)

x = Dense(256, activation='relu', name="Custom_Dense_ReLU")(x)

x = Dropout(0.5, name="Dropout_Regularization")(x)

outputs = Dense(1, activation='sigmoid', name="Output_Sigmoid")(x)

model = Model(inputs, outputs)


# 3.(Model Summary)

print("\n--- هيكل النموذج (Model Architecture) ---")
model.summary()
print("-----------------------------------------\n")

# 4(Optimization & Loss Function)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

 
# (Training with Early Stopping)
 
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print("جاري بدء عملية التدريب...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping]
)
