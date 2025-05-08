import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 1. Update model_stats with post-fine-tuning metrics
for scheme in model_stats:
    pre_acc = scheme_accuracies[scheme]
    post_acc = fine_tune_metrics[scheme]['Accuracy']
    
    pre_params = model_stats[scheme]['Num Params (Pre)']
    post_params = model_stats[scheme]['Num Params (Post)']

    model_stats[scheme]['Accuracy (Pre)'] = pre_acc
    model_stats[scheme]['Accuracy (Post)'] = post_acc
    model_stats[scheme]['Compactness (Pre)'] = pre_acc / pre_params
    model_stats[scheme]['Compactness (Post)'] = post_acc / post_params

# 2. Plot: Pre vs Post Fine-Tuning Accuracy
schemes = list(model_stats.keys())
pre_accuracies = [model_stats[s]['Accuracy (Pre)'] for s in schemes]
post_accuracies = [model_stats[s]['Accuracy (Post)'] for s in schemes]

x = np.arange(len(schemes))
bar_width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, pre_accuracies, width=bar_width, label='Pre Fine-Tuning', color='skyblue')
plt.bar(x + bar_width/2, post_accuracies, width=bar_width, label='Post Fine-Tuning', color='lightgreen')
plt.xticks(x, schemes)
plt.ylim(0, 1)
plt.ylabel('Test Accuracy')
plt.title('Pre vs Post Fine-Tuning Accuracy for SCANN Schemes')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 3. Plot: Compactness Pre vs Post
pre_compact = [model_stats[s]['Compactness (Pre)'] for s in schemes]
post_compact = [model_stats[s]['Compactness (Post)'] for s in schemes]

plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, pre_compact, width=bar_width, label='Pre', color='mediumpurple')
plt.bar(x + bar_width/2, post_compact, width=bar_width, label='Post', color='mediumseagreen')
plt.xticks(x, schemes)
plt.ylabel('Accuracy per Parameter')
plt.title('Compactness Score (Pre vs Post Fine-Tuning)')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 4. Plot: Sparsity
sparsities = [model_stats[s]['Sparsity (%)'] for s in schemes]
plt.figure(figsize=(6, 5))
plt.bar(schemes, sparsities, color='goldenrod')
plt.ylabel('Sparsity (%)')
plt.title('Sparsity of SCANN Schemes')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 5. Summary Table
df_summary = pd.DataFrame(model_stats).T[
    ['Accuracy (Pre)', 'Accuracy (Post)', 'Num Params (Pre)','Num Params (Post)', 'Model Size (MB)',
     'Sparsity (%)', 'Compactness (Pre)', 'Compactness (Post)']
]
display(df_summary.round(4))  # If in notebook
# Or to print:
print(df_summary.round(4).to_string())

# 6. Confusion Matrix, ROC, PR Curves per scheme
for scheme in schemes:
    cm = fine_tune_metrics[scheme]['Confusion Matrix']
    y_true = fine_tune_metrics[scheme]['Labels']
    y_prob = fine_tune_metrics[scheme]['Probs']

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Scheme {scheme}')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Scheme {scheme}')
    plt.legend()
    plt.grid()
    plt.show()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Scheme {scheme}')
    plt.grid()
    plt.show()
