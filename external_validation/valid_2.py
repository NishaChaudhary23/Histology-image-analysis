#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
# from support import get_all_roc_coordinates, plot_roc_curve, calculate_tpr_fpr
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

def plot_roc_curve(tpr, fpr, scatter = True, ax = None, conf_key = ['normal', 'osmf', 'oscc']):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")



os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P2/wdoscc', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P2/mdoscc','/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P2/pdoscc'] 
datapath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P2/all'
plotpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/plots/project_2/'

# NEED TO REWRITE THE DATA INPUT PART
if not os.path.exists(datapath):
    os.makedirs(datapath)

master_dataframe = pd.DataFrame()

for i in paths:
    items = os.listdir(i)
    # shuffling the list
    np.random.shuffle(items)
    # taking the first 1000 items
    items = items[:100]
    print(len(items))
    # adding the path to the items
    items = [os.path.join(datapath, item) for item in items]
    # creating a dataframe
    df = pd.DataFrame(items, columns=['filename'])
    # adding the label
    df['class'] = i.split('/')[-1]
    # appending the dataframe to the master dataframe
    master_dataframe = master_dataframe.append(df)

print(master_dataframe)
# shuffling the master dataframe
# master_dataframe = master_dataframe.sample(frac=1).reset_index(drop=True)
print(master_dataframe)
    

datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
# Test Data
test_generator = datagen_test.flow_from_dataframe(
        dataframe=master_dataframe,
        folder=datapath,
        target_size=(300, 300),
        class_mode='categorical',
        shuffle=False,
        validate_filenames=False,
        classes=['pdoscc', 'mdoscc', 'wdoscc'],)


conf_key = [*test_generator.class_indices.keys()]
print(conf_key)

# loading the model
model_2a = load_model('/home/chs.rintu/Documents/office/researchxoscc/Ensemble/models_available/M2a/finetune/M2a_fold_1_finetune.h5')
model_2b = load_model('/home/chs.rintu/Documents/office/researchxoscc/Ensemble/models_available/M2b/finetune/M2b_fold_3_finetune.h5')
# model.summary()
# model_2a.summary()
# model_2b.summary()

eval_2a = model_2a.predict(test_generator)
eval_2b = model_2b.predict(test_generator)
# finding the class with the highest probability
scores_2a = eval_2a
scores_2b = eval_2b

eval  = np.zeros((len(scores_2a), 3))
for i in range(len(eval)):
    eval[i][0] =scores_2a[i][0]
    eval[i][1] =scores_2a[i][1]*eval_2b[i][0]
    eval[i][2] =scores_2a[i][1]*eval_2b[i][1]
    print(eval[i])
    print(np.sum(eval[i]))

conf = confusion_matrix(test_generator.classes, np.argmax(eval, axis=1))

# conf = conf.values[:,1:]
conf = conf.astype(np.int32)
conf_percentages = conf / conf.sum(axis=1)[:, np.newaxis]
conf_percentages = conf_percentages * 100
conf_percentages = np.round(conf_percentages, 2).flatten()
labels = [f"{v1}\n{v2}%" for v1, v2 in
        zip(conf.flatten(),conf_percentages)]
labels = np.asarray(labels).reshape(3,3)
plt.figure(figsize=(3.5,3))
sns.heatmap(conf_percentages.reshape((3,3)), annot=labels, xticklabels=conf_key, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), yticklabels=conf_key, fmt='', cbar=True, annot_kws={"font":'Sans',"size": 9.5,"fontstyle":'italic' })
plt.xlabel('Predicted',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
plt.ylabel('Ground Truth',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
plt.title(f'External Validation Confusion Matrix',fontname="Sans", fontsize=11,fontweight='bold')
plt.tight_layout()
plt.savefig(f'{plotpath}project_1_exVal_cm.png', dpi = 300)

gt = np.array(test_generator.classes)
score_gt = np.array([eval[i][gt[i]] for i in range(len(gt))])
# gt = [conf_key[i] for i in gt]

X_test = pd.DataFrame(data={'class':gt, 'prob':score_gt})

bins = [i/20 for i in range(20)] + [1]
classes = [0, 1, 2]
roc_auc_ovr = {}
for i in range(len(classes)):
    # Gets the class
    c = classes[i]
    
    # Prepares an auxiliar dataframe to help with the plots
    df_aux = X_test.copy()
    df_aux['class'] = df_aux['class'].apply(lambda x: 1 if x == c else 0)
    df_aux = df_aux.reset_index(drop = True)
    
    # Plots the probability distribution for the class and the rest
    try:
        plt.figure(figsize = (3.5,3))
        ax = plt.subplot(1,1,1)
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(f'Probability Distribution for {conf_key[c]}')
        ax.legend([f"Class: {conf_key[c]}", "Rest"], loc = 'upper center')
        ax.set_xlabel(f"P(x = {conf_key[c]})")
        # Calculates the ROC Coordinates and plots the ROC Curves
        plt.tight_layout()  
        plt.savefig(f'{plotpath}project_1_exVal_probability_distribution_{conf_key[c]}.png', dpi = 300)
    except:
        print(f"Error in Probability Distribution for {conf_key[c]}")
        pass

    try:
        plt.figure(figsize = (3.5,3))
        ax_bottom = plt.subplot(1, 1, 1)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom, conf_key = ['normal', 'osmf', 'oscc'])
        ax_bottom.set_title(f'ROC Curve OvR for {conf_key[c]}')
        plt.tight_layout()  
        plt.savefig(f'{plotpath}project_1_exVal_roc_ovr_curve_{conf_key[c]}.png', dpi = 300)
    except:
        print(f"Error in ROC Curve for {conf_key[c]}")
        pass
    # Calculates the ROC AUC OvR
    roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'], multi_class = 'ovr')

print("Individual Figures Plot Complete")

plt.figure(figsize = (12, 8))
bins = [i/20 for i in range(20)] + [1]
classes = [0, 1, 2]
roc_auc_ovr = {}
try:
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]
        
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = X_test.copy()
        df_aux['class'] = df_aux['class'].apply(lambda x: 1 if x == c else 0)
        df_aux = df_aux.reset_index(drop = True)
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 3, i+1)
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(f'Figures for {conf_key[c]}')
        ax.legend([f"Class: {conf_key[c]}", "Rest"], loc = 'upper center')
        ax.set_xlabel(f"P(x = {conf_key[c]})")
        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 3, i+4)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom, conf_key = ['normal', 'osmf', 'oscc'])
        ax_bottom.set_title("ROC Curve OvR")
        
        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'], multi_class = 'ovr')
    plt.tight_layout()
    plt.savefig(f'{plotpath}project_1_exVal_roc.png', dpi = 300)
    print("Composite Figure Plot Complete")
except:
    print("Error in Composite Figure Plot")
    pass