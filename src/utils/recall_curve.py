import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc 
from sklearn.metrics import precision_recall_curve
import nrrd
import json


def compute_recall_curve(
    filename_predict: str,
    filename_truth: str,
    target_filename_plot: str,
    target_filename_auc: str
) -> None :

    if os.path.exists(target_filename_plot):
        print(f"{target_filename_plot} already exists, skipping")
        return
    
    data_predict = np.load(filename_predict)
    y_truth, header_truth = nrrd.read(filename_truth)

    y_predict = data_predict["softmax"][1]


    precision, recall, thresholds = precision_recall_curve(y_truth, y_predict)

    auc_score = auc(recall, precision)

    # Sauvegarde figure :

    os.makedirs(os.path.dirname(target_filename_plot), exist_ok=True)

    plt.plot(recall, precision, color = 'red', label = 'nnUnet', linewidth = 5)
    plt.plot(recall, recall, color = 'grey', linestyle = '--', linewidth = 4)
    plt.legend()
    plt.savefig(target_filename_plot, format = 'png')

    # Sauvegarde AUC :
    
    os.makedirs(os.path.dirname(target_filename_auc), exist_ok=True)
    with open(target_filename_auc, 'r') as f:
        dict = json.load(f)
    name = os.path.splitext(filename_predict)[0]
    dict[name] = auc_score
    with open(target_filename_auc, 'w') as f:
        json.dump(dict,f)
    






