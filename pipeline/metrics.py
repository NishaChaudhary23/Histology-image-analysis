import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
import numpy as np

basepath = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
metrics = pd.read_csv(os.path.join(basepath, 'test_pipeline_output.csv'))
# printing all incorrect predictions
# print(metrics[metrics['ground_truth'] != metrics['final_prediction']])
print(metrics.head(5))
# # setting confidence_2a if final_prediction is pdoscc or confidence_2a otherwis
# metrics_2a = metrics[['model_2a','confidence_2a','ground_truth']]
# metrics_2b = metrics[['model_2b','confidence_2b','ground_truth']]

# # changing ground truth to wmodoscc in metrics_2a if mdoscc or wdoscc is predicted
# metrics_2a['ground_truth'] = metrics_2a.apply(lambda x: 'wmdoscc' if x['ground_truth'] == 'mdoscc' or x['ground_truth'] == 'wdoscc' else x['ground_truth'], axis=1).astype(str)
# print (metrics_2a.head(5))
# print(metrics_2a[metrics_2a['ground_truth'] != metrics_2a['model_2a']])
# # filtering out all rows where ground truth is not pdoscc

# metrics_2a_0 = metrics_2a[metrics_2a['ground_truth'] != 'pdoscc']
# # Compute ROC curve and ROC area for each class
# fpr, tpr, _ = roc_curve(metrics_2a_0['model_2a'].apply(lambda x: 1 if x == 'pdoscc' else 0), metrics_2a_0['confidence_2a'], pos_label=0)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# # plt.show()


# # changing ground truth to wmodoscc in metrics_2a if mdoscc or wdoscc is predicted
# metrics_2a['ground_truth'] = metrics_2a.apply(lambda x: 'wmdoscc' if x['ground_truth'] == 'mdoscc' or x['ground_truth'] == 'wdoscc' else x['ground_truth'], axis=1).astype(str)
# print (metrics_2a.head(5))
# print(metrics_2a[metrics_2a['ground_truth'] != metrics_2a['model_2a']])
# # filtering out all rows where ground truth is not pdoscc

# metrics_2a_1 = metrics_2a[metrics_2a['ground_truth'] != 'pdoscc']
# # Compute ROC curve and ROC area for each class
# fpr, tpr, _ = roc_curve(metrics_2a_1['model_2a'].apply(lambda x: 1 if x == 'pdoscc' else 0), metrics_2a_1['confidence_2a'], pos_label=0)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# # plt.show()


metrics_final = metrics[['model_2a','confidence_2a','model_2b','confidence_2b','ground_truth']]
# inserting new column called final_confidence initialised to 0
metrics_final.insert(0, 'final_prediction', 0)
# calculating final confidence if model_2a is pdoscc then confidence_2a else confidence_2b
metrics_final['final_confidence'] = metrics_final.apply(lambda x: x['confidence_2a'] if x['model_2a'] == 'pdoscc' else x['confidence_2b'], axis=1).astype(float)
metrics_final = metrics_final[['final_prediction','final_confidence','ground_truth']]
print(metrics_final.head(5))
fpr, tpr, _ = roc_curve(metrics_final['final_prediction'].apply(lambda x: 2 if x == 'pdoscc' else 1 if x == 'mdoscc' else 0), metrics_final['final_confidence'], pos_label=0)
roc_auc = auc(fpr, tpr)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()