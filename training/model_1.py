#!/usr/bin/env python
# coding: utf-8

#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_large = '/storage/bic/data/oscc/project_1/train'


#reading & displaying an image
a = np.random.choice(['normal','osmf','oscc'])
path = '/storage/bic/data/oscc/project_1/train/{}/'.format(a)


#(trainX, testX, trainY, testY) = train_test_split(data, train_large.target, test_size=0.25)

# ImageDataGenerator
# color images
model_type = 'InceptionV3'
# model_type = 'DenseNet121'
print(f'Model Type: {model_type}_2_pool_dense_dense_1')

datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,vertical_flip=True)
# Training Data with Augmentation
train_generator = datagen_train.flow_from_directory(
	train_large,
	target_size=(300, 300),
	batch_size=32 if model_type not in ['NASNetLarge','NASNetMobile'] else 8,
	class_mode='categorical',
	subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_directory(
	train_large,
	target_size=(300, 300),
	batch_size=32 if model_type not in ['NASNetLarge','NASNetMobile'] else 8,
	class_mode='categorical',
	subset = 'validation',
	shuffle=False)

# finding class keys
class_keys = train_generator.class_indices.keys()
print(class_keys)
class_keys = valid_generator.class_indices.keys()
print(class_keys)


if model_type == 'InceptionV3':
	inception = InceptionV3(
			weights='imagenet',
			include_top=False,
			input_shape=(300,300,3)
			)
	for layer in inception.layers:
			layer.trainable = True
	x = layers.Flatten()(inception.output)
	# adding avgpool layer
	x = layers.GlobalAveragePooling2D()(inception.output)
	x = layers.Dense(1024, activation = 'relu', kernel_regularizer=l2(0.01))(x)
	x = layers.Dense(3, activation = 'softmax', kernel_regularizer=l2(0.01))(x)
	model = Model(inception.input, x)
	model.compile(optimizer = RMSprop(learning_rate = 0.0000001), loss = 'categorical_crossentropy', metrics = ['acc'])

if not os.path.exists(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1'):
	os.makedirs(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1')


#TF_CPP_MIN_LOG_LEVEL=2
# Training the model

print("------------------------------------------")
print(f'Training the model {model_type}_2_pool_dense_dense_1')
print("------------------------------------------")
filepath = f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/model_log'
if os.path.exists(filepath):
        os.makedirs(filepath)
filepath = filepath + "/model-{epoch:02d}-{val_acc:.2f}.h5"
callbacks = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
history = model.fit(train_generator, validation_data = valid_generator, verbose=1, epochs=20, callbacks=callbacks)

print("------------------------------------------")
print(f'Training Complete')
print("------------------------------------------")
# Creating a directory to save the model paths 

# Saving the model
model.save(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/{model_type}_2_pool_dense_dense_1.h5')
print("------------------------------------------")
print(f'Model saved')
print("------------------------------------------")


#plotting the accuracy and loss
print("------------------------------------------")
print(f'Plotting and supplimentary data')
print("------------------------------------------")
plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/Accuracy.jpg')

#np.save('/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/history1.npy',history.history)

hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/history.json' 
with open(hist_json_file, mode='w') as f:
	hist_df.to_json(f)

# or save to csv: 
hist_csv_file = f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/history.csv'
with open(hist_csv_file, mode='w') as f:
	hist_df.to_csv(f)
	
loaded_model = load_model(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/{model_type}_2_pool_dense_dense_1.h5')
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
plt.savefig(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/Confusion_matrix.jpg')

conf_df = pd.DataFrame(confusion, index = ['normal','osmf','oscc'], columns = ['normal','osmf','oscc'])
conf_df.to_csv(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/Confusion_matrix.csv')

# classification report
target_names = ['normal','osmf','oscc']
report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'/storage/bic/data/oscc/project_1/models/{model_type}_2_pool_dense_dense_1/Classification_report.csv')

print("------------------------------------------")
print(f'Supplimentary Data Saved')
print("------------------------------------------")
