# 1. LIBRARIES 
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from preprocessing import X_train, X_test, y_train, y_test, RANDOM_STATE
from feature_selection import main_feature_pipeline
from evaluation import evaluate_model

# 2. MODEL DEFINITIONS 
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
                    max_iter=2000, random_state=RANDOM_STATE
                ), 
                cv=5, 
                n_jobs=-1
            ))
        ])
    }


# 3. TRAIN AND COMPARE ENSEMBLE MODELS 
ensemble_models = build_ensemble_models(main_feature_pipeline)
ensemble_results = []

for name, model in ensemble_models.items():
    result = evaluate_model(
        model, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        model_name=name
    )
    ensemble_results.append(result)

ensemble_results_df = pd.DataFrame([
    {k: v for k, v in res.items() if k not in ["y_pred", "y_prob", "fitted_model"]
    }
    for res in ensemble_results
])

print("Ensemble model comparison:")
print(ensemble_results_df.sort_values(
    by="F1 Score", 
    ascending=False).round(4))