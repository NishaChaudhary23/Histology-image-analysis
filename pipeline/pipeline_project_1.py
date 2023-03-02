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

def phase(choice):
        choice_d = ''
        base = '/home/chs.rintu/Documents/office/researchxoscc/project_1/dataSet'
        out_path = '/home/chs.rintu/Documents/office/researchxoscc/project_1/output'
        data_src_path = f'{base}/pipeline'
        datapath = f'{base}/train_all'
        if choice=='1':
                label_1 = 'normal'
                label_2 = 'osmf'
                label_3 = 'oscc'
                class_names = [label_1, label_2, label_3]

        for i in range(5):
                print("------------------------------------------------------------------------------------")
                print("------------------------------------------")
                print(f'Fold {i+1}')
                print("------------------------------------------")
                print("------------------------------------------------------------------------------------")
                
                df_test = pd.read_csv(f'{data_src_path}/data_fold_{i}/internal_val.csv')
                df_train = pd.read_csv(f'{data_src_path}/data_fold_{i}/train.csv')
                print(df_train[df_train['filename'].isin(df_test['filename'])])
                # Training Data
                datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.25)
                train_generator = datagen_train.flow_from_dataframe(
                        dataframe=df_train,
                        folder=datapath,
                        target_size=(300, 300),
                        batch_size=32,
                        class_mode='categorical',
                        subset='training',
                        validate_filenames=False)
                valid_generator = datagen_train.flow_from_dataframe(
                        dataframe=df_train,
                        folder=datapath,
                        target_size=(300, 300),
                        batch_size=32,
                        class_mode='categorical',
                        subset='validation',
                        shuffle=True,
                        validate_filenames=False)

                datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
                # Test Data
                test_generator = datagen_test.flow_from_dataframe(
                        dataframe=df_test,
                        folder=datapath,
                        target_size=(300, 300),
                        class_mode='categorical',
                        shuffle=True,
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
                x = layers.Dense(3, activation = 'softmax')(x)
                model = Model(inception.input, x)
                model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])


                if not os.path.exists(f'{out_path}/model_1'):
                        os.makedirs(f'{out_path}/model_1')
                if not os.path.exists(f'{out_path}/model_1/fold_{i+1}'):
                        os.makedirs(f'{out_path}/model_1/fold_{i+1}')
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
                model.save(f'{out_path}/model_1/fold_{i+1}/{choice}.h5')
                print("------------------------------------------")
                print(f'Model saved')
                print("------------------------------------------")


                #plotting the accuracy and loss
                print("------------------------------------------")
                print(f'Plotting and supplimentary data')
                print("------------------------------------------")
                plt.figure(figsize=(3.5,3))
                plt.plot(history.history['acc'], label='Train Acc')
                plt.plot(history.history['val_acc'], label='Val Acc')
                plt.xlabel('Epochs',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.ylabel('Accuracy',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.title(f'Accuracy for {choice}',fontname="Sans", fontsize=11,fontweight='bold')
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/Accuracy.jpg')
                plt.figure(figsize=(3.5,3))
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.xlabel('Epochs',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.ylabel('Loss',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.title(f'Loss for {choice}',fontname="Sans", fontsize=11,fontweight='bold')
                plt.legend(loc = 0)
                plt.tight_layout()
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/Loss.jpg')

                # np.save('{out_path}/model_1/history1.npy',history.history)

                hist_df = pd.DataFrame(history.history) 

                # save to json:  
                hist_json_file = f'{out_path}/model_1/fold_{i+1}/history.json' 
                with open(hist_json_file, mode='w') as f:
                        hist_df.to_json(f)

                # or save to csv: 
                hist_csv_file = f'{out_path}/model_1/fold_{i+1}/history.csv'
                with open(hist_csv_file, mode='w') as f:
                        hist_df.to_csv(f)

                loaded_model = load_model(f'{out_path}/model_1/fold_{i+1}/{choice}.h5')
                outcomes = loaded_model.predict(valid_generator)
                y_pred = np.argmax(outcomes, axis=1)
                # confusion matrix
                confusion = confusion_matrix(valid_generator.classes, y_pred)
                conf_df = pd.DataFrame(confusion)
                conf_df.to_csv(f'{out_path}/model_1/fold_{i+1}/train_time_test_confusion_matrix.csv')
                conf = pd.read_csv(f'{out_path}/model_1/fold_{i+1}/train_time_test_confusion_matrix.csv')
                conf = conf.values[:,1:]
                conf = conf.astype(np.int32)
                conf_percentages = conf / conf.sum(axis=1)[:, np.newaxis]
                conf_percentages = conf_percentages * 100
                conf_percentages = np.round(conf_percentages, 2).flatten()
                print(conf_percentages)
                labels = [f"{v1}\n{v2}%" for v1, v2 in
                        zip(conf.flatten(),conf_percentages)]
                labels = np.asarray(labels).reshape((2,2))
                print(labels)
                plt.figure(figsize=(3.5,3))
                sns.heatmap(conf_percentages.reshape((3,3)), annot=labels, xticklabels=class_names, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), yticklabels=class_names, fmt='', cbar=True, annot_kws={"font":'Sans',"size": 9.5,"fontstyle":'italic' })
                plt.xlabel('Predicted',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.ylabel('Ground Truth',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.title(f'Confusion Matrix for {choice}',fontname="Sans", fontsize=11,fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/train_time_test_confusion_matrix.jpg')

                # classification report
                report = classification_report(valid_generator.classes, y_pred, output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv(f'{out_path}/model_1/fold_{i+1}/train_time_classification_report.csv')

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
                y_pred = model.predict(test_generator)
                y_pred = np.argmax(y_pred, axis=1)
                confusion = confusion_matrix(test_generator.classes, y_pred)
                plt.figure(figsize=(10, 10))
                sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels = class_names, yticklabels = class_names)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/internal_validation_confusion_matrix.jpg')

                conf_df = pd.DataFrame(confusion)
                conf_df.to_csv(f'{out_path}/model_1/fold_{i+1}/internal_validation_confusion_matrix.csv')

                # classification report

                report = classification_report(test_generator.classes, y_pred, output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv(f'{out_path}/model_1/fold_{i+1}/internal_validation_classification_report.csv')

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
                model.save(f'{out_path}/model_1/fold_{i+1}/{choice}_finetune.h5')
                print("------------------------------------------")
                print(f'Model saved')
                print("------------------------------------------")


                #plotting the accuracy and loss
                print("------------------------------------------")
                print(f'Plotting and supplimentary data')
                print("------------------------------------------")
                plt.figure(figsize=(3.5,3))
                plt.plot(history.history['acc'], label='Train Loss')
                plt.plot(history.history['val_acc'], label='Val Loss')
                plt.xlabel('Epochs',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.ylabel('Accuracy',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.title(f'Accuracy for {choice}',fontname="Sans", fontsize=11,fontweight='bold')
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/Accuracy_finetune.jpg')
                plt.figure(figsize=(3.5,3))
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Val Loss')
                plt.xlabel('Epochs',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.ylabel('Loss',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.title(f'Loss for {choice}',fontname="Sans", fontsize=11,fontweight='bold')
                plt.legend(loc = 0)
                plt.tight_layout()
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/Loss_finetune.jpg')

                # np.save('{out_path}/model_1/history1.npy',history.history)

                hist_df = pd.DataFrame(history.history) 

                # save to json:  
                hist_json_file = f'{out_path}/model_1/fold_{i+1}/history_finetune.json' 
                with open(hist_json_file, mode='w') as f:
                        hist_df.to_json(f)

                # or save to csv: 
                hist_csv_file = f'{out_path}/model_1/fold_{i+1}/history_finetune.csv'
                with open(hist_csv_file, mode='w') as f:
                        hist_df.to_csv(f)

                loaded_model = load_model(f'{out_path}/model_1/fold_{i+1}/{choice}.h5')
                outcomes = loaded_model.predict(valid_generator)
                test_fine_pred = np.argmax(outcomes, axis=1)
                # confusion matrix
                confusion = confusion_matrix(valid_generator.classes, test_fine_pred)
                plt.figure(figsize=(10, 10))
                sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels = class_names, yticklabels = class_names)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/train_time_test_confusion_matrix_finetune.jpg')

                conf_df = pd.DataFrame(confusion)
                conf_df.to_csv(f'{out_path}/model_1/fold_{i+1}/train_time_test_confusion_matrix_finetune.csv')

                # classification report
                report = classification_report(valid_generator.classes, test_fine_pred, output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv(f'{out_path}/model_1/fold_{i+1}/train_time_test_classification_report_finetune.csv')

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
                y_pred = model.predict(test_generator)
                y_pred = np.argmax(y_pred, axis=1)
                confusion = confusion_matrix(test_generator.classes, y_pred)

                conf_df = pd.DataFrame(confusion)
                conf_df.to_csv(f'{out_path}/model_1/fold_{i+1}/internal_validation_confusion_matrix_finetune.csv')
                conf = pd.read_csv(f'{out_path}/model_1/fold_{i+1}/train_time_test_confusion_matrix.csv')
                conf = conf.values[:,1:]
                conf = conf.astype(np.int32)
                conf_percentages = conf / conf.sum(axis=1)[:, np.newaxis]
                conf_percentages = conf_percentages * 100
                conf_percentages = np.round(conf_percentages, 2).flatten()

                print(conf_percentages)
                labels = [f"{v1}\n{v2}%" for v1, v2 in
                        zip(conf.flatten(),conf_percentages)]
                labels = np.asarray(labels).reshape((3,3))
                print(labels)
                plt.figure(figsize=(3.5,3))
                sns.heatmap(conf_percentages.reshape((3,3)), annot=labels, xticklabels=class_names, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), yticklabels=class_names, fmt='', cbar=True, annot_kws={"font":'Sans',"size": 9.5,"fontstyle":'italic' })
                plt.xlabel('Predicted',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.ylabel('Ground Truth',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
                plt.title(f'Confusion Matrix for {choice}',fontname="Sans", fontsize=11,fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{out_path}/model_1/fold_{i+1}/internal_validation_confusion_matrix_finetune.jpg')

                # classification report

                report = classification_report(test_generator.classes, y_pred, output_dict=True)
                df = pd.DataFrame(report).transpose()
                df.to_csv(f'{out_path}/model_1/fold_{i+1}/internal_validation_classification_report_finetune.csv')

                print("------------------------------------------")
                print(f'Supplimentary Testing Phase Data Saved locally')
                print("------------------------------------------")
                print("------------------------------------------------------------------------------------")



phase('1')