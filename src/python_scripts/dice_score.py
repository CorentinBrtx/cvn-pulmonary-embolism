import os
from argparse import ArgumentParser
from nnUNet.nnunet.evaluation.evaluator import aggregate_scores


parser = ArgumentParser(description="Compute DICE scores for files in two folders (ground_truth and predicted)")

parser.add_argument("gt_folder", help="Ground Truth directory name")
parser.add_argument("pred_folder", help="Predictions directory name")
parser.add_argument("-d", "--destination", default="dice.out", help="Name of the file to output the DICE score")
parser.add_argument("-v", action="store_true", help="Verbosity")
args = parser.parse_args()

pred_gt_pairs = []

for ground_truth in os.listdir(args.gt_folder):
    file_name = os.path.basename(ground_truth)
    predicted = os.path.join(args.pred_folder, file_name)
    if os.path.exists(predicted):
        pred_gt_pairs.append((predicted, os.path.join(args.gt_folder, ground_truth)))
    elif args.v:
        print(f"No prediction for {file_name}")

output_file = open(args.destination, "w")
aggregate_scores(pred_gt_pairs, json_output_file=output_file)
output_file.close()

if args.v:
    print(f"Wrote scores at {args.destination}")

    