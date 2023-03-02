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
