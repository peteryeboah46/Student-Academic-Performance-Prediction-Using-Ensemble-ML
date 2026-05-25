# 1. LIBRARIES 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from preprocessing import X_train, X_test, y_train
from ensemble_models import ensemble_results, ensemble_results_df

# 2. SHAP EXPLAINABILITY 
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

# 2.1 SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values.values, X_test_shap, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig("Saved_images/9_Shap_summary_plot.png", 
            dpi=300, bbox_inches="tight")
plt.show()

# 2.2 SHAP Waterfall Plot for Student 1
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("Saved_images/10_Shap_waterfall_student1.png", 
            dpi=300, bbox_inches="tight")
plt.show()

# 2.3 SHAP Dependence Plot for Top Feature
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