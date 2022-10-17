import pandas as pd
from collections import defaultdict
import sklearn.metrics

gold_reports = pd.read_csv('../chexpert-results/mimic_cxr_caption.csv')
pred_reports = pd.read_csv('../chexpert-results/mimic_cxr_pred.csv')
cols = list(pred_reports.columns[2:])
print(len(cols))

id2label = {}
for idx in range(len(gold_reports)):
    l = gold_reports.iloc[idx]
    id = l['image_ids']
    id2label[id] = l

total_pred = defaultdict(list)
total_gold = defaultdict(list)

for idx in range(len(pred_reports)):
    # pred
    l1 = pred_reports.iloc[idx]
    id = l1['image_ids']
    # gold
    l2 = id2label[id]
    
    label1 = l1[2:]
    label2 = l2[2:]
    
    i = 0
    for x,y in zip(label1, label2):
        if x == 1 or x == -1:
            xx = 1
        else:
            xx = 0
        if y == 1 or y == -1:
            yy = 1
        else:
            yy = 0
        total_pred[cols[i]].append(xx)
        total_gold[cols[i]].append(yy)
        i += 1

scores = []
precisions = []
recalls = []
f1s = []
accs = []
for k in cols:
    pred = total_pred[k]
    gold = total_gold[k]
    acc = [x == y for x, y in zip(pred, gold)]
    acc = sum(acc) / len(acc)
    accs.append(acc)
    score = sklearn.metrics.precision_recall_fscore_support(gold, pred, average='binary')
    scores.append(score)
    precisions.append(score[0])
    recalls.append(score[1])
    f1s.append(score[2])

print('Accuracy', sum(accs) / len(accs))
print('Precision', sum(precisions) / len(precisions))
print('Recall', sum(recalls) / len(recalls))
print('F-1', sum(f1s) / len(f1s))





























