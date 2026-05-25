# 0. LIBRARIES
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    RFE,
    SelectFromModel
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import df, X_train, X_test, y_train, y_test, preprocessor, RANDOM_STATE
from evaluation import evaluate_model


# 1. FEATURE SELECTION 
# 1.1 Correlation with G3 (interpretation only)
corr_with_target = (df.select_dtypes(include=[np.number])
                    .corr()["G3"].sort_values(ascending=False))
print("Correlation of numerical features with G3:")
print(corr_with_target.round(4))

# 1.2 Define feature-selection pipelines
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

# 1.3 Extract selected features
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

# 1.4 Build feature-selection comparison matrix
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

# 2. COMPARE FEATURE SELECTION METHODS 
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