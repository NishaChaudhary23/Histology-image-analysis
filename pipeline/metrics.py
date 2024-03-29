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


metrics_final = metrics
# inserting new column called final_confidence initialised to 0
metrics_final.insert(0, 'final_confidence', 0)
# calculating final confidence if model_2a is pdoscc then confidence_2a else confidence_2b
metrics_final['final_confidence'] = metrics_final.apply(lambda x: x['confidence_2a'] if x['model_2a'] == 'pdoscc' else x['confidence_2b'], axis=1).astype(float)
metrics_final = metrics_final[['final_prediction','final_confidence','ground_truth']]
# converting final prediction to 0 if wdoscc, 1 if mdoscc, 2 if pdoscc
metrics_final['final_prediction'] = metrics_final.apply(lambda x: 2 if x['final_prediction'] == 'pdoscc' else 1 if x['final_prediction'] == 'mdoscc' else 0, axis=1).astype(int)
print(metrics_final.head(5))



# for class 0
metrics_final_0 = metrics_final[metrics_final['ground_truth'] == 'wdoscc']
print(metrics_final_0.head(5))
fpr_0, tpr_0, _ = roc_curve(metrics_final_0['final_prediction'], metrics_final_0['final_confidence'], pos_label=0)
roc_auc_0 = auc(fpr_0, tpr_0)
plt.figure()
plt.plot(fpr_0, tpr_0, color='darkorange', lw=2, label=f'ROC curve wdoscc (area = %0.2f)' % roc_auc_0)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(basepath, 'roc_curve_wdoscc.png'))
plt.show()



# for class 1
metrics_final_1 = metrics_final[metrics_final['ground_truth'] == 'mdoscc']
print(metrics_final_1.head(5))
fpr_1, tpr_1, _ = roc_curve(metrics_final_1['final_prediction'], metrics_final_1['final_confidence'], pos_label=1)
roc_auc_1 = auc(fpr_1, tpr_1)
plt.figure()
plt.plot(fpr_1, tpr_1, color='red', lw=2, label=f'ROC curve mdoscc (area = %0.2f)' % roc_auc_1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# for class 2
metrics_final_2 = metrics_final[metrics_final['ground_truth'] == 'pdoscc']
print(metrics_final_2.head(5))
fpr_2, tpr_2, _ = roc_curve(metrics_final_2['final_prediction'], metrics_final_2['final_confidence'], pos_label=2)
roc_auc_2 = auc(fpr_2, tpr_2)
plt.figure()
plt.plot(fpr_2, tpr_2, color='green', lw=2, label=f'ROC curve pdoscc (area = %0.2f)' % roc_auc_2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()