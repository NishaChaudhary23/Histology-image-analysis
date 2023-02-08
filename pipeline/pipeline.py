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
from sklearn.preprocessing import OneHotEncoder
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
        base = '/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet'
        out_path = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
        datapath = f'{base}/train_all'
        if choice=='M1a':
                df_train = pd.read_csv(f'{base}/pipeline/pw_m/train.csv')
                df_test = pd.read_csv(f'{base}/pipeline/pw_m/test.csv')
                label_1 = 'wpdoscc'
                label_2 = 'mdoscc'
        if choice=='M1b':
                df_train = pd.read_csv(f'{base}/pipeline/p_w/train.csv')
                df_test = pd.read_csv(f'{base}/pipeline/p_w/test.csv')
                label_1 = 'pdoscc'
                label_2 = 'wdoscc'
        if choice=='M2a':
                df_train = pd.read_csv(f'{base}/pipeline/wm_p/train.csv')
                df_test = pd.read_csv(f'{base}/pipeline/wm_p/test.csv')
                label_1 = 'wpdoscc'
                label_2 = 'mdoscc'
        if choice=='M2b':
                df_train = pd.read_csv(f'{base}/pipeline/w_m/train.csv')
                df_test = pd.read_csv(f'{base}/pipeline/w_m/test.csv')
                label_1 = 'pdoscc'
                label_2 = 'wdoscc'

        # remapping the col namesas x_lab and y_lab
        df_train = df_train.rename(columns={'image':'filename'})
        df_train = df_train.rename(columns={'label':'class'})
        df_test = df_test.rename(columns={'image':'filename'})
        df_test = df_test.rename(columns={'label':'class'})



        # converting to string in dataframe
        df_train['class'] = df_train['class'].astype(str)
        df_test['class'] = df_test['class'].astype(str)
        print(df_train)
        print(df_test)


        # 0s and 1s to label key 
        df_train['class'] = df_train['class'].replace({str(0):label_1, str(1):label_2})
        df_test['class'] = df_test['class'].replace({str(0):label_1, str(1):label_2})

        print(df_train)
        print(df_test)
        
        
        # Training Data
        datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2)
        train_generator = datagen_train.flow_from_dataframe(
                dataframe=df_train,
                folder=datapath,
                target_size=(300, 300),
                batch_size=32,
                class_mode='categorical',
                subset = 'training',
                validate_filenames=False)
        #Validation Data
        valid_generator = datagen_train.flow_from_dataframe(
                dataframe=df_train,
                folder=datapath,
                target_size=(300, 300),
                batch_size=32,
                class_mode='categorical',
                subset = 'validation',
                shuffle=False,
                validate_filenames=False)

        datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
        # Test Data
        test_generator = datagen_test.flow_from_dataframe(
                dataframe=df_test,
                folder=datapath,
                target_size=(300, 300),
                class_mode='categorical',
                shuffle=False,
                validate_filenames=False)
        
        # printing the train, valid and test data
        print("------------------------------------------")
        print(f'Training Data: {train_generator.n}')
        print(f'Validation Data: {valid_generator.n}')
        print(f'Test Data: {test_generator.n}')
        print("------------------------------------------")

        
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


        TF_CPP_MIN_LOG_LEVEL=2
        # Training the model

        print("------------------------------------------")
        print(f'Training the model {choice}')
        print("------------------------------------------")
        history = model.fit(train_generator, validation_data = valid_generator, epochs=20)

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
        test_finetune_test_pred = model.predict(test_generator)
        test_finetune_test_pred = np.argmax(test_finetune_test_pred, axis=1)
        confusion = confusion_matrix(test_generator.classes, test_finetune_test_pred)
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

        report = classification_report(test_generator.classes, test_finetune_test_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(f'{out_path}/{choice}/Classification_report_test.csv')

        print("------------------------------------------")
        print(f'Supplimentary Testing Phase Data Saved')
        print("------------------------------------------")

        # finetuning the model
        print("------------------------------------------")
        print(f'Finetuning the model')
        print("------------------------------------------")
        # freezing the base model
        inception.trainable = False
        # recompiling the model
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
        # training the model
        history = model.fit(train_generator, validation_data = valid_generator, epochs=20)
        print("------------------------------------------")
        print(f'Finetuning Complete')
        print("------------------------------------------")
        # Creating a directory to save the model paths 

        # Saving the model
        model.save(f'{out_path}/{choice}/{choice}_finetune.h5')
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
        plt.savefig(f'{out_path}/{choice}/Accuracy_finetune.jpg')

        # np.save('{out_path}/{choice}/history1.npy',history.history)

        hist_df = pd.DataFrame(history.history) 

        # save to json:  
        hist_json_file = f'{out_path}/{choice}/history_finetune.json' 
        with open(hist_json_file, mode='w') as f:
                hist_df.to_json(f)

        # or save to csv: 
        hist_csv_file = f'{out_path}/{choice}/history_finetune.csv'
        with open(hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)

        loaded_model = load_model(f'{out_path}/{choice}/{choice}.h5')
        outcomes = loaded_model.predict(valid_generator)
        test_fine_pred = np.argmax(outcomes, axis=1)
        # confusion matrix
        confusion = confusion_matrix(valid_generator.classes, test_fine_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'{out_path}/{choice}/Confusion_matrix_finetune.jpg')

        conf_df = pd.DataFrame(confusion)
        conf_df.to_csv(f'{out_path}/{choice}/Confusion_matrix_finetune.csv')

        # classification report
        report = classification_report(valid_generator.classes, test_fine_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(f'{out_path}/{choice}/Classification_report_finetune.csv')

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
        test_finetune_test_pred = model.predict(test_generator)
        test_finetune_test_pred = np.argmax(test_finetune_test_pred, axis=1)
        confusion = confusion_matrix(test_generator.classes, test_finetune_test_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'{out_path}/{choice}/Confusion_matrix_test_finetune.jpg')

        conf_df = pd.DataFrame(confusion)
        conf_df.to_csv(f'{out_path}/{choice}/Confusion_matrix_test_finetune.csv')

        # classification report

        report = classification_report(test_generator.classes, test_finetune_test_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(f'{out_path}/{choice}/Classification_report_test_finetune.csv')

        print("------------------------------------------")
        print(f'Supplimentary Testing Phase Data Saved')
        print("------------------------------------------")



phase('M2b')