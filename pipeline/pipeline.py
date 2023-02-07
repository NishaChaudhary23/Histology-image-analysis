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
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_large = '/storage/bic/data/oscc/data/working/train'


#reading & displaying an image
a = np.random.choice(['wdoscc','mdoscc','pdoscc'])
path = '/storage/bic/data/oscc/data/working/train/{}/'.format(a)

def phase(choice):
        out_path = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
        datapath = '/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/train_all'
        if choice=='M1a':
                df_train = pd.read_csv('/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/pipeline/pw_m/train.csv')
                df_test = pd.read_csv('/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/pipeline/pw_m/test.csv')
        if choice=='M1b':
                df_train = pd.read_csv('/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/pipeline/p_w/train.csv')
                df_test = pd.read_csv('/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/pipeline/p_w/test.csv')
        datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.20)

        # remapping the col namesas x_lab and y_lab
        df_train = df_train.rename(columns={'image':'filename'})
        df_train = df_train.rename(columns={'label':'class'})
        df_test = df_test.rename(columns={'image':'filename'})
        df_test = df_test.rename(columns={'label':'class'})

        # converting all to categorical
        df_train['class'] = to_categorical(df_train['class'], num_classes=2, dtype='str')
        df_test['class'] = to_categorical(df_test['class'], num_classes=2, dtype='str')

        print(df_train.head())
        print(df_test.head())
        # Training Data
        train_generator = datagen_train.flow_from_dataframe(
                dataframe=df_train,
                folder=datapath,
                target_size=(300, 300),
                batch_size=32,
                class_mode='categorical',
                subset = 'training')
        #Validation Data
        valid_generator = datagen_train.flow_from_dataframe(
                dataframe=df_train,
                folder=datapath,
                target_size=(300, 300),
                batch_size=32,
                class_mode='categorical',
                subset = 'validation',
                shuffle=False)

        datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
        # Test Data
        test_generator = datagen_test.flow_from_dataframe(
                dataframe=df_test,
                folder=datapath,
                target_size=(300, 300),
                batch_size=32,
                class_mode='categorical',
                shuffle=False)

        
        inception = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in inception.layers:
                layer.trainable = True
        x = layers.Flatten()(inception.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(2, activation = 'softmax')(x)
        model = Model(inception.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])


        if not os.path.exists(f'{out_path}/{choice}'):
                os.makedirs(f'{out_path}/{choice}')
        # Model Summary


        #TF_CPP_MIN_LOG_LEVEL=2
        # Training the model

        print("------------------------------------------")
        print(f'Training the model {choice}')
        print("------------------------------------------")
        history = model.fit(train_generator, validation_data = valid_generator, epochs=50)

        print("------------------------------------------")
        print(f'Training Complete')
        print("------------------------------------------")
        # Creating a directory to save the model paths 

        # Saving the model
        model.save(f'{out_path}/{choice}/{choice}.h5')
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
        plt.savefig(f'{out_path}/{choice}/Accuracy.jpg')

        # np.save('{out_path}/{choice}/history1.npy',history.history)

        hist_df = pd.DataFrame(history.history) 

        # save to json:  
        hist_json_file = f'{out_path}/{choice}/history.json' 
        with open(hist_json_file, mode='w') as f:
                hist_df.to_json(f)

        # or save to csv: 
        hist_csv_file = f'{out_path}/{choice}/history.csv'
        with open(hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)

        loaded_model = load_model(f'{out_path}/{choice}/{choice}.h5')
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
        plt.savefig(f'{out_path}/{choice}/Confusion_matrix.jpg')

        conf_df = pd.DataFrame(confusion)
        conf_df.to_csv(f'{out_path}/{choice}/Confusion_matrix.csv')

        # classification report
        report = classification_report(valid_generator.classes, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(f'{out_path}/{choice}/Classification_report.csv')

        print("------------------------------------------")
        print(f'Supplimentary Data Saved')
        print("------------------------------------------")

        # Testing the model
        print("------------------------------------------")
        print(f'Testing the model')
        print("------------------------------------------")
        test_loss, test_acc = model.evaluate(test_generator)
        print('test acc:', test_acc)
        print('test loss:', test_loss)

        # Predicting the test data
        print("------------------------------------------")
        print(f'Predicting the test data')
        print("------------------------------------------")
        test_pred = model.predict(test_generator)
        test_pred = np.argmax(test_pred, axis=1)
        confusion = confusion_matrix(test_generator.classes, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'{out_path}/{choice}/Confusion_matrix_test.jpg')

        conf_df = pd.DataFrame(confusion)
        conf_df.to_csv(f'{out_path}/{choice}/Confusion_matrix_test.csv')

        # classification report

        report = classification_report(test_generator.classes, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(f'{out_path}/{choice}/Classification_report_test.csv')

        print("------------------------------------------")
        print(f'Supplimentary Testing Phase Data Saved')
        print("------------------------------------------")


phase('M1a')