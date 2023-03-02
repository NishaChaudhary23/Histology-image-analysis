import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
import numpy as np

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
# encoding the ground_truth column
metrics['ground_truth'] = metrics['ground_truth'].map({'wdoscc': 0, 'mdoscc': 1, 'pdoscc': 2})
print(metrics.head(5))

#  calculating the accuracy
y_true = np.array(metrics['ground_truth'].values)
y_scores = np.array(metrics['confidence'].values)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true[:]==i, y_scores[:], pos_label=i)

for i in range(3):
    roc_auc[i] = auc(fpr[i], tpr[i])

print(fpr)
print(tpr)
print(roc_auc)

# plot ROC curves for each class and macro-average
plt.figure()
lw = 2
colors = ['red', 'green', 'blue', 'deeppink']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()