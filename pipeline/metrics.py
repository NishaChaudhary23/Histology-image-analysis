import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
import numpy as np

basepath = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
metrics = pd.read_csv(os.path.join(basepath, 'test_pipeline_output.csv'))
# printing all incorrect predictions
print(metrics[metrics['ground_truth'] != metrics['final_prediction']])
print(metrics.head(5))
# setting confidence_2a if final_prediction is pdoscc or confidence_2a otherwis
metrics_2a = metrics[['model_2a','confidence_2a','ground_truth']]
metrics_2b = metrics[['model_2b','confidence_2b','ground_truth']]

# changing ground truth to wmodoscc in metrics_2a if mdoscc or wdoscc is predicted
metrics_2a['ground_truth'] = metrics_2a.apply(lambda x: 'wmdoscc' if x['ground_truth'] == 'mdoscc' or x['ground_truth'] == 'wdoscc' else x['ground_truth'], axis=1).astype(str)
print (metrics_2a.head(5))
# if ground truth doesnt equal to model_2a then confidence_2a is 1-confidence_2a
metrics_2a['confidence_2a'] = metrics_2a.apply(lambda x: 1-x['confidence_2a'] if x['ground_truth'] != x['model_2a'] else x['confidence_2a'], axis=1).astype(float)
print (metrics_2a.head(5))
print(metrics_2a['ground_truth'] != metrics_2a['model_2a'])


# # Compute ROC curve and ROC area for each class
# fpr, tpr, _ = roc_curve(metrics_2a['ground_truth'].apply(lambda x: 1 if x == 'pdoscc' else 0), metrics_2a['confidence_2a'])
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
# plt.show()