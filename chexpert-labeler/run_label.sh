PATH1='../results/mimic_cxr_weighted_pred.json'
PATH2='../chexpert-results/mimic_cxr_weighted_pred.csv'

python label_on_fly.py --reports_path $PATH1 --output_path $PATH2