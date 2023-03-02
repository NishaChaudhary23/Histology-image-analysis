import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
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
accuracy = metrics[metrics['prediction'] == metrics['ground_truth']].shape[0] / metrics.shape[0]
print(f'Accuracy: {accuracy}')
# Calculating the AUC score based on the final confidence for all three classes present
auc_score = roc_auc_score(pd.get_dummies(metrics['ground_truth']), pd.get_dummies(metrics['prediction']), multi_class='ovr')
print(f'AUC score: {auc_score}')
# calculating the ROC curve for all three classes present
RocCurveDisplay.from_predictions(metrics['ground_truth'], pd.get_dummies(metrics['prediction']))
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true==i, y_scores[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plot ROC curves for each class and micro-average
plt.figure()
lw = 2
colors = ['red', 'green', 'blue', 'deeppink']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot(fpr["micro"], tpr["micro"], color='gold', lw=lw,
         label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()