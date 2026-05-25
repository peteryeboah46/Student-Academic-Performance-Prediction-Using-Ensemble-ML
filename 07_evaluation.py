# 1. LIBRARIES 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from preprocessing import X, y, X_train, X_test, y_train, y_test, RANDOM_STATE, output_dir
from feature_selection import main_feature_pipeline
from individual_models import build_base_models, base_results, base_results_df
from ensemble_models import build_ensemble_models, ensemble_results, ensemble_results_df


# 2. CROSS-VALIDATION SETUP 
cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 3. EVALUATION FUNCTION 
def evaluate_model(model, X_train, X_test, y_train, y_test, 
                   model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = (roc_auc_score(
        y_test, y_prob) if y_prob is not None else np.nan)

    return {"Model": model_name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1 Score": f1,"ROC-AUC": roc,"y_pred": y_pred,
        "y_prob": y_prob, "fitted_model": model}


# 4. COMBINE ALL RESULTS 
all_results_df = pd.concat([base_results_df, ensemble_results_df], 
                           ignore_index=True)

print("All model results:")
print(all_results_df.sort_values(
    by="F1 Score", ascending=False).round(4))

# 5. BAR CHART COMPARISON (2 x 3) 
metrics_to_plot = (
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])

# CREATE MODEL NAME MAPPING 
model_map = {"Logistic Regression": "LR", "Decision Tree": "DT",
    "Random Forest": "RF", "SVM": "SVM", "Bagging": "Bag",
    "AdaBoost": "Ada", "Gradient Boosting": "GB",
    "Stacking": "Stack"}

# Order: Individual models first, then ensemble models
model_order = (
    ["LR", "DT", "RF", "SVM", "Bag", "Ada", "GB", "Stack"])

# Create a copy so original dataframe is not altered
plot_df = all_results_df.copy()
plot_df["Model"] = plot_df["Model"].map(model_map)

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

# ONE MAIN HEADING
fig.suptitle(
    "Performance Comparison of Individual and \n
    Ensemble Models Across Evaluation Metrics",
    fontsize=30,fontweight="bold", y=0.93)

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]

    sns.barplot(
    data=plot_df, x="Model", y=metric, order=model_order, ax=ax)

    ax.set_title(metric, fontsize=16, fontweight="bold", pad=8)
    ax.set_xlabel("Model", fontsize=20, fontweight="bold")
    ax.set_ylabel(metric, fontsize=20, fontweight="bold")
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=11)

# Add model values vertically inside bars 
    for p in ax.patches:
        height = p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = height * 0.5

        text = ax.text(x, y, f"{height:.3f}", ha="center",
            va="center", rotation=90, fontsize=25,
            color="white", fontweight="bold")
        
# Delete empty subplot
fig.delaxes(axes[5])

# ADD BOXED ABBREVIATION NOTE INSIDE THE FIGURE
fig.text(0.5, 0.015,
    "LR = Logistic Regression; DT = Decision Tree; RF = Random Forest;"
    "SVM = Support Vector Machine; Bag = Bagging; Ada = AdaBoost; "
    "GB = Gradient Boosting; Stack = Stacking.", ha="center",
    va="center", fontsize=16, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
              edgecolor="black"))

plt.tight_layout(rect=[0, 0.06, 1, 0.94])
plt.subplots_adjust(wspace=0.35, hspace=0.35)

plt.savefig(
"Saved_images/5_Model_Bar_Chart_Comparison_Combined_2x3.png",
dpi=300, bbox_inches="tight")

plt.show()

# 6. CROSS-VALIDATION FOR ALL MODELS 
scoring = {"accuracy": "accuracy","precision": "precision",
    "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

# Define target 
y_target = df["performance"]

cv_results = []

all_models = {}
all_models.update(build_base_models(main_feature_pipeline))
all_models.update(build_ensemble_models(main_feature_pipeline))

for name, model in all_models.items(): scores = cross_validate(
        model, X, y_target, cv=cv, scoring=scoring, n_jobs=-1)

    cv_results.append({"Model": name,
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Precision": scores["test_precision"].mean(),
        "CV Recall": scores["test_recall"].mean(),
        "CV F1": scores["test_f1"].mean(),
        "CV ROC-AUC": scores["test_roc_auc"].mean()})

cv_results_df = pd.DataFrame(cv_results)
cv_results_df = cv_results_df.sort_values(
by="CV F1", ascending=False)

print("Cross-validation results:")
print(cv_results_df.round(4))

# 7. ROC CURVES AND CONFUSION MATRICES 
# 7.1 Individual Models
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle(
"ROC Curves and Confusion Matrices of Individual Models", 
fontsize=25)
    
for i, res in enumerate(base_results):
# ROC
    ax1 = axes[0, i]
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_title(
    res["Model"], fontsize=20, fontweight="bold", pad=20)
    ax1.set_xlabel("False Positive Rate", fontsize=20)
    ax1.set_ylabel("True Positive Rate", fontsize=20)
    ax1.legend(loc="lower right", fontsize=20)
    ax1.tick_params(axis="both", labelsize=12)

# Confusion Matrix
    ax2 = axes[1, i]
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title(
    f'{res["Model"]}', fontsize=20, fontweight="bold", pad=20)
    ax2.set_xlabel("Predicted label", fontsize=20)
    ax2.set_ylabel("True label", fontsize=20)
    ax2.tick_params(axis="both", labelsize=12)

# Adjust confusion matrix cell number size
    for text in axes[1, i].texts:
        text.set_fontsize(25)
        text.set_fontweight("bold")

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
plt.subplots_adjust(wspace=0.8, hspace=0.7)

plt.savefig("Saved_images/6_ROC_Curves_and_CM_of_IM.png", 
            dpi=300, bbox_inches="tight")
plt.show()

# 7.2 Ensemble Models
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle(
"ROC Curves and Confusion Matrices of Ensemble Models", 
fontsize=25)

for i, res in enumerate(ensemble_results):
    # ROC
    ax1 = axes[0, i]
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_title(
    res["Model"], fontsize=20, fontweight="bold", pad=20)
    ax1.set_xlabel("False Positive Rate", fontsize=20)
    ax1.set_ylabel("True Positive Rate", fontsize=20)
    ax1.legend(loc="lower right", fontsize=20)
    ax1.tick_params(axis="both", labelsize=12)

# Confusion Matrix
    ax2 = axes[1, i]
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title(
    f'{res["Model"]}', fontsize=20, fontweight="bold", pad=20)
    ax2.set_xlabel("Predicted label", fontsize=20)
    ax2.set_ylabel("True label", fontsize=20)
    ax2.tick_params(axis="both", labelsize=12)

# Adjust confusion matrix cell number size
    for text in axes[1, i].texts:
        text.set_fontsize(25)
        text.set_fontweight("bold")

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
plt.subplots_adjust(wspace=0.8, hspace=0.7)

plt.savefig("Saved_images/7_ROC_Curves_and_CM_of_EM.png", 
            dpi=300, bbox_inches="tight")
plt.show()

# 7.3 SINGLE COMBINED FIGURE: ROC + CM (2 x 8 layout)
fig, axes = plt.subplots(2, 8, figsize=(30, 10))
fig.suptitle("ROC Curves and Confusion Matrices for \n
Individual and Ensemble Models",
             fontsize=40, fontweight="bold", y=1.03)

# Main headers
fig.text(0.25, 0.9, "ROC Curves", ha="center", 
fontsize=25, fontweight="bold")
fig.text(0.75, 0.9, "Confusion Matrices", ha="center", 
fontsize=25, fontweight="bold")

# Row labels
fig.text(0.02, 0.72, "Individual Models", va="center", 
rotation=90,
         fontsize=22, fontweight="bold")
fig.text(
0.02, 0.28, "Ensemble Models", va="center", rotation=90, 
fontsize=22, fontweight="bold")

# Top row: Individual models
for i, res in enumerate(base_results):
# ROC CURVES
    ax_roc = axes[0, i]
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, linewidth=2, 
    label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    ax_roc.set_title(res["Model"], fontsize=20, 
    fontweight="bold", pad=20)
    ax_roc.set_xlabel("FPR", fontsize=20, fontweight="bold")
    ax_roc.set_ylabel("TPR", fontsize=20, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=20)
    ax_roc.tick_params(axis="both", labelsize=12)

# CONFUSION MATRIX
    ax_cm = axes[0, i + 4]
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title(
    res["Model"],fontsize=20,fontweight="bold", pad=20)
    ax_cm.set_xlabel("Predicted", fontsize=20, fontweight="bold")
    ax_cm.set_ylabel("Actual", fontsize=20, fontweight="bold")
    ax_cm.tick_params(axis="both", labelsize=12)

# Adjust confusion matrix cell number size
    for text in axes[0, i+4].texts:
        text.set_fontsize(25)
        text.set_fontweight("bold")

# Bottom row: Ensemble models
for i, res in enumerate(ensemble_results):
# ROC CURVES
    ax_roc = axes[1, i]
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(
    fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    ax_roc.set_title(
    res["Model"],fontsize=20,fontweight="bold", pad=20)
    ax_roc.set_xlabel("FPR", fontsize=20, fontweight="bold")
    ax_roc.set_ylabel("TPR", fontsize=20, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=20)
    ax_roc.tick_params(axis="both", labelsize=12)

# CONFUSION MATRIX
    ax_cm = axes[1, i + 4]
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title(
    res["Model"], fontsize=20, fontweight="bold", pad=20)
    ax_cm.set_xlabel(
    "Predicted", fontsize=20, fontweight="bold")
    ax_cm.set_ylabel("Actual", fontsize=20, fontweight="bold")
    ax_cm.tick_params(axis="both", labelsize=12)

# Adjust confusion matrix cell number size
    for text in axes[1, i+4].texts:
        text.set_fontsize(25)
        text.set_fontweight("bold")

plt.tight_layout(rect=[0.04, 0.04, 0.99, 0.92])
plt.subplots_adjust(wspace=0.8, hspace=0.7)

plt.savefig("Saved_images/8_ROC_Curves_CM_for_IM_and_EM.png",
    dpi=300, bbox_inches="tight")
plt.show()

#ROC CURVE AND CONFUSION MATRIX FOR ONLY OVERALL BEST-PERFORMING MODEL
# ROC CURVE FOR GRADIENT BOOSTING ONLY
for res in ensemble_results:
    if res["Model"] == "Gradient Boosting" and res["y_prob"] is not None:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Gradient Boosting")
        plt.legend(loc="lower right")
        plt.tight_layout()

        plt.savefig(
        "Saved_images/6_ROC_Ensemble_Gradient_Boosting.png",
        dpi=300, bbox_inches="tight")
        plt.show()

# CONFUSION MATRIX FOR GRADIENT BOOSTING ONLY
for res in ensemble_results:
    if res["Model"] == "Gradient Boosting":
        cm = confusion_matrix(y_test, res["y_pred"])

        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format="d", colorbar=False)

        plt.title("Confusion Matrix - Gradient Boosting")
        plt.tight_layout()

        plt.savefig(
        "Saved_images/7_C_Matrix_Ensemble_Gradient_Boosting.png",
        dpi=300, bbox_inches="tight")
        plt.show()