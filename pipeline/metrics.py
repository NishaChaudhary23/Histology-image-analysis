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

metrics_2a = metrics["model_2a","confidence_2a","ground_truth"]
metrics_2b = metrics["model_2b","confidence_2b","ground_truth"]

# changing ground truth to wmodoscc in metrics_2a if mdoscc or wdoscc is predicted
metrics_2a['ground_truth'] = metrics_2a.apply(lambda x: 'wmodoscc' if x['ground_truth'] == 'mdoscc' or x['ground_truth'] == 'wdoscc' else x['ground_truth'], axis=1).astype(str)
print (metrics_2a.head(5))