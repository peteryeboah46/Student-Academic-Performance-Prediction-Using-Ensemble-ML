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

# 3. LOAD DATASET 
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

# 4. CREATE TARGET VARIABLE 
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

# 5. DEFINE FEATURES AND SPLIT DATA
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