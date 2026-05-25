# 1. LIBRARIES 
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from preprocessing import X_train, X_test, y_train, y_test, RANDOM_STATE
from feature_selection import main_feature_pipeline
from evaluation import evaluate_model

# 2. MODEL DEFINITIONS 
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

# 3. TRAIN AND COMPARE INDIVIDUAL MODELS 
base_models = build_base_models(main_feature_pipeline)
base_results = []

for name, model in base_models.items():
    result = evaluate_model(
        model, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        model_name=name
    )
    base_results.append(result)

base_results_df = pd.DataFrame([
    {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", "fitted_model"]
    }
    for res in base_results
])

print("Individual model comparison:")
print(base_results_df.sort_values(by="F1 Score", ascending=False).round(4))