def evaluate_model(sdnet, X_test, y_test, scheme="A", save_dir="figures"):
    """Evaluate a trained SDNN model and generate plots."""
    os.makedirs(save_dir, exist_ok=True)

    X_test = np.array(X_test)
    y_test = np.where(np.array(y_test) == 'Stress', 1, 0) if isinstance(y_test[0], str) else np.array(y_test)

    sdnet.loadData(mode='test', X_test=X_test, y_test=y_test)

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in sdnet.testloader:
            inputs, labels = data
            outputs = sdnet.forward(inputs, retain_grad=False)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n[SCANN-{scheme}] Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Relax", "Stress"], yticklabels=["Relax", "Stress"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - Scheme {scheme}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{scheme}.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Scheme {scheme}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"roc_curve_{scheme}.png"))
    plt.close()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - Scheme {scheme}")
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"pr_curve_{scheme}.png"))
    plt.close()

    # Optionally compute sparsity
    sparsity = compute_sparsity(sdnet)

    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc_score,
        'Confusion Matrix': cm,
        'Sparsity (%)': sparsity
    }
