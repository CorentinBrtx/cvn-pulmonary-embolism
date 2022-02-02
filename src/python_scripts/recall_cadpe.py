import os

from src.utils.recall_curve import compute_recall_curve

#######################################################################
# Compute recall on CAD_PE dataset

WORKDIR = os.getenv("WORKDIR")
path_truth = "/workdir/shared/pulmembol/nnUNet/nnUNet_raw/nnUNet_raw_data/Task501_EmbolismCADPE/labelsTr"

path_pred = "/workdir/shared/pulmembol/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task501_EmbolismCADPE"

path_res = "/workdir/shared/pulmembol/nnUNet/nnUNet_results/Task501_EmbolismCADPE/res_auc/auc_score.json"
data_folder = int(os.getenv("SLURM_ARRAY_TASK_ID_PRED"))

folds = [folder for folder in os.listdir(data_folder) if folder.startswith("fold")]

files_truth = os.listdir(path_truth)

for fold in folds:
    files = os.listdir(os.path.join(fold, "validation_raw/"))
    for file_pred in files:
        if file_pred.endswith(".npz"):
            print(f"Processing {file_pred}")
            list_files_truth = [file for file in files_truth if file.endswith(".nii.gz") and os.path.basename(file) == os.path.basename(file_pred)]
            if (len(list_files_truth) == 0) or (len(list_files_truth) > 1):
                print("Ground truth invalid.")
            else:
                file_truth = list_files_truth[0]
                compute_recall_curve (
                    filename_predict = file_pred,
                    filename_truth = file_truth,
                    target_filename_plot = os.path.splitext(file_pred)[0] + "_plot_recall.png",
                    target_filename_auc = path_res
                )


