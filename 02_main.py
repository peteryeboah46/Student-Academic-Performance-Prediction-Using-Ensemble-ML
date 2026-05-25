# 1. LIBRARIES 
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, 
    RFE, SelectFromModel)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
    BaggingClassifier,AdaBoostClassifier,
    GradientBoostingClassifier,StackingClassifier)

from sklearn.metrics import (accuracy_score,
    precision_score, recall_score,f1_score,
    roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc)

import shap

# 2. SETTINGS 
set_config(display="diagram")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 6)

output_dir = "Saved_images"

# . LOAD DATASET 
df = pd.read_csv("student-por.csv", sep=";")

print("First 5 rows:")
print(df.head())

print("Dataset shape:")
print(df.shape)

print("Column names:")
print(df.columns.tolist())

print("Data types:")
print(df.dtypes)

print("Missing values:")
print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())

print("Summary statistics:")
print(df.describe().round(4).T)

# 4. EXPLORATORY DATA ANALYSIS 

# 4.1 Distribution of Final Grade (G3)
plt.figure(figsize=(8, 6))
sns.histplot(df["G3"], bins=20, kde=True)
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Saved_images/1_Distribution_of_Final_Grade_(G3).png", 
            dpi=300)
plt.show()

# 4.2 Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(
    numeric_df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig(
    "Saved_images/2_Correlation_heatmap_of_Numerical_Variables.png",
    dpi=300)
plt.show()

# 4.3 Boxplots of Selected Numerical Variables
selected_boxplot_vars = ["absences", "studytime", "G1", "G2", "G3"]
fig, ax = plt.subplots(figsize=(10, 6))
box = ax.boxplot(
    [df[col].dropna() for col in selected_boxplot_vars],
    patch_artist=True,
    labels=selected_boxplot_vars
)

colors = ["lightblue", "lightgreen", "lightcoral", "plum", "khaki"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

for median in box["medians"]:
    median.set_color("red")
    median.set_linewidth(2)

plt.title("Boxplots of Selected Numerical Variables")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    "Saved_images/3_Boxplots_of_Selected_Numerical_Variables.png", 
    dpi=300)
plt.show()

# 5. CREATE TARGET VARIABLE 
# 1 = Pass if G3 >= 10 and # 0 = Fail / At-risk if G3 < 10
df["performance"] = np.where(df["G3"] >= 10, 1, 0)

print("Class Distribution:")
print(df["performance"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="performance", data=df, palette=["red", "green"])
plt.title("Class Distribution: Student Performance")
plt.xlabel("Performance (0 = Fail, 1 = Pass)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    "Saved_images/4_Class_Distribution_Student_Performance.png", 
    dpi=300)
plt.show()

# 6. DEFINE FEATURES AND SPLIT DATA
X = df.drop(columns=["performance", "G3"])
y = df["performance"]

print("Predictor matrix shape:", X.shape)
print("Target vector shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y,
    random_state=RANDOM_STATE)

print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# 7. PRE-PROCESSING PIPELINES 
categorical_features = (
    X_train.select_dtypes(include=["object"]).columns.tolist())
numerical_features = (
    X_train.select_dtypes(exclude=["object"]).columns.tolist())

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

numeric_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

categorical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_preprocessor, numerical_features),
    ("cat", categorical_preprocessor, categorical_features)])

# 8. FEATURE SELECTION 
# 8.1 Correlation with G3 (interpretation only)
corr_with_target = (df.select_dtypes(include=[np.number])
                    .corr()["G3"].sort_values(ascending=False))
print("Correlation of numerical features with G3:")
print(corr_with_target.round(4))

# 8.2 Define feature-selection pipelines
feature_pipeline_mutual_info = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(
        score_func=mutual_info_classif, k=20))])

rfe_estimator = LogisticRegression(max_iter=2000,
    class_weight="balanced", random_state=RANDOM_STATE)

feature_pipeline_rfe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("selector", RFE(
        estimator=rfe_estimator, n_features_to_select=20))])

embedded_estimator = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE)

feature_pipeline_embedded = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("selector", SelectFromModel(estimator=embedded_estimator, 
                                 threshold="median"))])
feature_pipelines = {
    "Mutual Info": feature_pipeline_mutual_info,
    "RFE": feature_pipeline_rfe,
    "Embedded": feature_pipeline_embedded}

# 8.3 Extract selected features
selected_features_by_method = {}

for fs_name, fs_pipeline in feature_pipelines.items():
    fs_pipeline.fit(X_train, y_train)

    preprocessor_step = fs_pipeline.named_steps["preprocessor"]
    selector = fs_pipeline.named_steps["selector"]

    feature_names = preprocessor_step.get_feature_names_out()
    support_mask = selector.get_support()
    selected_features = feature_names[support_mask].tolist()

    selected_features_by_method[fs_name] = selected_features

    print(f"\nSelected features using {fs_name}:")
    print(selected_features)

# 8.4 Build feature-selection comparison matrix
all_selected_features = sorted(set(sum(
    selected_features_by_method.values(), [])))

comparison_data = []
for feature in all_selected_features:
    row =({"Feature": feature, "Mutual Info":(
        "Yes" if feature in selected_features_by_method["Mutual Info"] else "No"),
    "RFE": (
        "Yes" if feature in selected_features_by_method["RFE"] else "No"),
    "Embedded": (
        "Yes" if feature in selected_features_by_method["Embedded"] else "No")
})
    yes_count = ((row["Mutual Info"] == "Yes") +
        (row["RFE"] == "Yes") + (row["Embedded"] == "Yes"))
    row["Final Selection"] = "Yes" if yes_count >= 2 else "No"
    comparison_data.append(row)

feature_selection_matrix = pd.DataFrame(comparison_data)
print("Feature Selection Comparison Matrix:")
print(feature_selection_matrix)

# 9. MODEL DEFINITIONS 
def build_base_models(feature_pipeline):
    return {
        "Logistic Regression": Pipeline(steps=[
            ("features", feature_pipeline),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE))]),
        "Decision Tree": Pipeline(steps=[
            ("features", feature_pipeline),
            ("classifier", DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                class_weight="balanced",
                random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline(steps=[
            ("features", feature_pipeline),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                class_weight="balanced",
                random_state=RANDOM_STATE))]),
        "SVM": Pipeline(steps=[
            ("features", feature_pipeline),
            ("classifier", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE))])}

def build_ensemble_models(feature_pipeline):
    base_lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", 
        random_state=RANDOM_STATE)
    base_dt = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, class_weight="balanced", 
        random_state=RANDOM_STATE)
    base_rf = RandomForestClassifier(
        n_estimators=300, min_samples_split=5, class_weight="balanced", 
        random_state=RANDOM_STATE)
    base_svm = SVC(
        kernel="rbf", probability=True, class_weight="balanced", 
        random_state=RANDOM_STATE)

    return {
        "Bagging": Pipeline(steps=[("features", feature_pipeline),
            ("classifier", BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    random_state=RANDOM_STATE), n_estimators=100,
                random_state=RANDOM_STATE))]),
        "AdaBoost": Pipeline(steps=[
            ("features", feature_pipeline),
            ("classifier", AdaBoostClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline(steps=[
            ("features", feature_pipeline),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=RANDOM_STATE))]),
        "Stacking": Pipeline(steps=[
            ("features", feature_pipeline),
            ("classifier", StackingClassifier(
                estimators=[
                    ("lr", base_lr),
                    ("dt", base_dt),
                    ("rf", base_rf),
                    ("svm", base_svm)],
                final_estimator=LogisticRegression(
                    max_iter=2000, random_state=RANDOM_STATE), 
                cv=5, n_jobs=-1))])}

# 10. CROSS-VALIDATION SETUP 
cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 11. EVALUATION FUNCTION 
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

# 12. COMPARE FEATURE SELECTION METHODS 
feature_selection_results = []

for fs_name, fs_pipeline in feature_pipelines.items():
    model = Pipeline(
        steps=[("features", fs_pipeline),(
            "classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", 
                random_state=RANDOM_STATE))])

    result = evaluate_model(model, X_train, X_test, y_train, y_test,
        model_name=f"Logistic Regression + {fs_name}")

    feature_selection_results.append({"Feature Selection": fs_name,
        "Accuracy": result["Accuracy"],"Precision": result["Precision"],
        "Recall": result["Recall"], "F1 Score": result["F1 Score"],
        "ROC-AUC": result["ROC-AUC"]})

feature_selection_df = pd.DataFrame(feature_selection_results)
feature_selection_df = feature_selection_df.sort_values(
    by="F1 Score", ascending=False)

print("Feature selection comparison:")
print(feature_selection_df.round(4))

best_fs_name = (
    feature_selection_df.iloc[0]["Feature Selection"])
print(f"Best feature selection method: {best_fs_name}")

main_feature_pipeline = feature_pipelines[best_fs_name]

# 13. TRAIN AND COMPARE INDIVIDUAL MODELS 
base_models = build_base_models(main_feature_pipeline)
base_results = []

for name, model in base_models.items():
    result = evaluate_model(
        model, X_train, X_test, y_train, y_test, model_name=name)
    base_results.append(result)

base_results_df = pd.DataFrame([
    {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", 
                                               "fitted_model"]}
    for res in base_results])

print("Individual model comparison:")
print(base_results_df.sort_values(by="F1 Score", 
                                  ascending=False).round(4))

# 14. TRAIN AND COMPARE ENSEMBLE MODELS 
ensemble_models = build_ensemble_models(main_feature_pipeline)
ensemble_results = []

for name, model in ensemble_models.items():
    result = evaluate_model(
        model, X_train, X_test, y_train, y_test, model_name=name)
    ensemble_results.append(result)

ensemble_results_df = pd.DataFrame([
    {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", 
                                               "fitted_model"]}
    for res in ensemble_results])

print("Ensemble model comparison:")
print(ensemble_results_df.sort_values(
    by="F1 Score", ascending=False).round(4))

# 15. COMBINE ALL RESULTS 
all_results_df = pd.concat([base_results_df, ensemble_results_df], 
                           ignore_index=True)

print("All model results:")
print(all_results_df.sort_values(
    by="F1 Score", ascending=False).round(4))

# 16. BAR CHART COMPARISON (2 x 3) 
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

# 17. CROSS-VALIDATION FOR ALL MODELS 
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

# 18. ROC CURVES AND CONFUSION MATRICES 
# 18.1 Individual Models
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

# 18.2 Ensemble Models
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

# 18.3 SINGLE COMBINED FIGURE: ROC + CM (2 x 8 layout)
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

# 19. SHAP EXPLAINABILITY 
# Select best-performing model for SHAP analysis
best_model_name = ensemble_results_df.sort_values(
by="F1 Score", ascending=False).iloc[0]["Model"]
best_model_result = next(
res for res in ensemble_results if res["Model"] == best_model_name)
best_model = best_model_result["fitted_model"]

print(f"Best model selected for SHAP analysis: {best_model_name}")

# Transform test data to feature space used by model
feature_transformer = best_model.named_steps["features"]
X_train_transformed = feature_transformer.fit_transform(X_train, y_train)
X_test_transformed = feature_transformer.transform(X_test)

# Convert to dense if sparse
if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

# Get feature names BEFORE selection
preprocessor = feature_transformer.named_steps["preprocessor"]
all_feature_names = preprocessor.get_feature_names_out()

# Get selector and keep only selected feature names
selector = feature_transformer.named_steps["selector"]
selected_feature_names = all_feature_names[selector.get_support()]

# Build dataframe with the correct number of selected columns
X_test_shap = pd.DataFrame(
X_test_transformed, columns=selected_feature_names)

# Access classifier
classifier = best_model.named_steps["classifier"]

# Build SHAP explainer
explainer = shap.Explainer(classifier.predict, X_test_shap)
shap_values = explainer(X_test_shap)

# 19.1 SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values.values, X_test_shap, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig("Saved_images/9_Shap_summary_plot.png", 
            dpi=300, bbox_inches="tight")
plt.show()

# 19.2 SHAP Waterfall Plot for Student 1
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("Saved_images/10_Shap_waterfall_student1.png", 
            dpi=300, bbox_inches="tight")
plt.show()

# 19.3 SHAP Dependence Plot for Top Feature
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_feature_idx = np.argmax(mean_abs_shap)
top_feature_name = selected_feature_names[top_feature_idx]

plt.figure()
shap.dependence_plot(top_feature_name,
    shap_values.values, X_test_shap, show=False)

plt.title(f"SHAP Dependence Plot for {top_feature_name}")
plt.tight_layout()
plt.savefig("Saved_images/11_Shap_dependence_plot.png", 
            dpi=300, bbox_inches="tight")
plt.show()