import os
import pandas as pd

basepath = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
metrics = pd.read_csv(os.path.join(basepath, 'test_pipeline_output.csv'))
print(metrics.head(5))
# inserting new column called final confidence to the original dataframe
metrics.insert(0, 'final_confidence', 0)
# setting confidence_2a if final_prediction is pdoscc or confidence_2a otherwise
metrics['final_confidence'] = metrics.apply(lambda x: x['confidence_2a'] if x['final_prediction'] == 'pdoscc' else x['confidence_2b'], axis=1).astype(float)
# Chosing only final_prediction and final_confidence and ground_truth
metrics = metrics[['final_prediction', 'final_confidence', 'ground_truth']]
# renaming the columns
metrics.columns = ['prediction', 'confidence', 'ground_truth']
print(metrics.head(5))

#  calculating the accuracy
accuracy = metrics[metrics['prediction'] == metrics['ground_truth']].shape[0] / metrics.shape[0]
print(f'Accuracy: {accuracy}')
# calculating the AUC score and curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
fpr, tpr, thresholds = roc_curve(metrics['ground_truth'], metrics['confidence'])
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc}')
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',  lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
