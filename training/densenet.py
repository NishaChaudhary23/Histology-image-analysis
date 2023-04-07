#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.models import load_model
import os
import cv2
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import kl_divergence
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.metrics import poisson

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_large = '/storage/bic/data/oscc/data/working/train'


#reading & displaying an image
a = np.random.choice(['wdoscc','mdoscc','pdoscc'])
path = '/storage/bic/data/oscc/data/working/train/{}/'.format(a)


#(trainX, testX, trainY, testY) = train_test_split(data, train_large.target, test_size=0.25)

# models = ['DenseNet121','DenseNet169','DenseNet201','MobileNetV3Large','MobileNetV3Small','EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7','InceptionResNetV2''InceptionV3','MobileNetV2','NASNetLarge','NASNetMobile','ResNet101','ResNet101V2','ResNet152','ResNet152V2','ResNet50','ResNet50V2','VGG16','VGG19','Xception']
# ImageDataGenerator
# color images

model_type = 'DenseNet121'

datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2)
# Training Data
train_generator = datagen_train.flow_from_directory(
        train_large,
        target_size=(300, 300),
        batch_size=100,
        class_mode='categorical',
        subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_directory(
        train_large,
        target_size=(300, 300),
        batch_size=100,
        class_mode='categorical',
        subset = 'validation',
        shuffle=False)


# Creating the model
if model_type == 'DenseNet121':
        densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(300,300,3)
        )
        for layer in densenet.layers:
                layer.trainable = True
        x = layers.Flatten()(densenet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(densenet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])


if not os.path.exists(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}'):
        os.makedirs(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}')
# Model Summary


#TF_CPP_MIN_LOG_LEVEL=2
# Training the model

print("------------------------------------------")
print(f'Training the model {model_type}')
print("------------------------------------------")
history = model.fit(train_generator, validation_data = valid_generator, epochs=50)

print("------------------------------------------")
print(f'Training Complete')
print("------------------------------------------")
# Creating a directory to save the model paths 

# Saving the model
model.save(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/dense121_01.h5')
print("------------------------------------------")
print(f'Model saved')
print("------------------------------------------")


#plotting the accuracy and loss
print("------------------------------------------")
print(f'Plotting and supplimentary data')
print("------------------------------------------")
plt.figure(figsize=(10, 10))
plt.lineplot(history.history['acc'], label='Training Accuracy')
plt.lineplot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Accuracy.jpg')

loaded_model = load_model(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/dense121_01.h5')
outcomes = loaded_model.predict(valid_generator)
y_pred = np.argmax(outcomes, axis=1)
# confusion matrix
confusion = confusion_matrix(valid_generator.classes, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Confusion_matrix.jpg')

conf_df = pd.DataFrame(confusion, index = ['wdoscc','mdoscc','pdoscc'], columns = ['wdoscc','mdoscc','pdoscc'])
conf_df.to_csv(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Confusion_matrix.csv')

# classification report
target_names = ['wdoscc','mdoscc','pdoscc']
report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Classification_report.csv')

# Other metrics
kldiv = kl_divergence(valid_generator.classes, y_pred)
mse = mean_squared_error(valid_generator.classes, y_pred)
pois = poisson(valid_generator.classes, y_pred)

with open(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Other_metrics.txt', 'w+') as f:
        f.write(f'KLD: {str(kldiv)}\n')
        f.write(f'MSE: {str(mse)}\n')
        f.write(f'POISSON: {str(pois)}\n')

print("------------------------------------------")
print(f'Supplimentary Data Saved')
print("------------------------------------------")