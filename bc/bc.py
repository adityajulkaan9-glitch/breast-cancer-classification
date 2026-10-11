# Step 1 — Breast Cancer Dataset

# 1. Data load karo

import pandas as pd

df = pd.read_csv("breast_cancer.csv")

# 2. Dataset ka size dekho
print(df.shape)

# 3. Pehli 5 rows dekho
print(df.head())

# 4. Column names dekho
print(df.columns.tolist())

# 5. Data types aur information dekho
df.info()
print(df.dtypes)


# 6. Basic statistics dekho
print(df.describe())

# 7. Missing values check karo
print(df.isnull().sum())

# 8. Duplicate rows check karo
print(df.duplicated().sum())

# 9. Target distribution check karo
print(df["diagnosis"].value_counts())
print(df["target"].value_counts())









# Step 2: EDA — 4 tasks
# # 1. Diagnosis ka distribution

# print(df["diagnosis"].value_counts())


# # 2. Bar chart banao

# import matplotlib.pyplot as plt

# df["diagnosis"].value_counts().plot(kind="bar")
# plt.show()


# # 3. Dono classes ka average radius compare karo
# print(df.groupby("diagnosis")["mean radius"].mean())


# # 4. Radius ka distribution dekho

# df.boxplot(column="mean radius", by="diagnosis")
# plt.show()


from ydata_profiling import ProfileReport

report = ProfileReport(df, title="Breast Cancer EDA")
report.to_file("eda_report.html")



# step 2 mein html main ye check kro 



# STEP 1: OVERVIEW
# HTML mein: Overview -> Overview tab
#
# Kya check karna hai?
# 1. Number of observations = kitni rows/patients hain?
# 2. Number of variables = kitne columns/features hain?
# 3. Missing cells = kitni values missing hain?
# 4. Duplicate rows = kya same row repeat ho rahi hai?
# 5. Variable types = columns numeric hain ya categorical?
#
# Kyun?
# Dataset ka size aur basic quality samajhne ke liye.
#
# Observation:
# Mere dataset mein 569 rows aur 32 columns hain.
# Missing cells = 0
# Duplicate rows = 0
# Note: Clean dikhne ka matlab ye nahi ki data mein koi
# aur problem nahi ho sakti.


# STEP 2: ALERTS
# HTML mein: Overview -> Alerts tab
#
# Kya check karna hai?
# 1. High correlation alerts
# 2. Constant ya almost-constant variables ke alerts
# 3. Unusual values ya doosre data-quality alerts
#
# Kyun?
# Report humein batati hai ki kin cheezon ko investigate
# karna chahiye.
#
# Important:
# Alert ka matlab automatically error nahi hota.
# Pehle reason samjho, phir action decide karo.
# Abhi koi feature delete nahi karna.


# STEP 3: VARIABLES
# HTML mein: Variables dropdown
#
# Pehle ye variables check karo:
# 1. diagnosis
# 2. target
# 3. mean radius
# 4. mean area
# 5. worst radius
#
# Har variable mein kya dekhna hai?
# - Distinct/Unique: Kitni alag values hain?
# - Missing: Kya koi value missing hai?
# - Mean/Median: Typical value kya hai?
# - Min/Max: Values kis range mein hain?
# - Histogram: Values ka distribution kaisa hai?
#
# Kyun?
# Har feature ki values aur behaviour samajhne ke liye.
#
# Important:
# diagnosis aur target same outcome ko represent karte hain.
# Model banate waqt target predict karenge aur diagnosis ko
# input feature mein nahi rakhenge.


# STEP 4: INTERACTIONS
# HTML mein: Interactions
#
# In pairs ko check karo:
# 1. mean radius vs mean area
# 2. mean radius vs mean perimeter
#
# Kya observe karna hai?
# - Kya ek feature badhne par doosra bhi badhta hai?
# - Scatter plot mein points kis pattern mein hain?
# - Kya koi point baaki points se kaafi alag hai?
#
# Kyun?
# Do features ke beech relationship samajhne ke liye.
#
# Important:
# Relationship dikhne ka matlab ye nahi ki koi feature
# useless hai. Abhi kuch remove nahi karna.


# STEP 5: CORRELATIONS
# HTML mein: Correlations
#
# Kya check karna hai?
# - Strong positive correlation
# - Strong negative correlation
# - Weak linear correlation
# - mean radius, mean perimeter aur mean area ka relation
#
# Correlation ko kaise samjhein?
# +1 ke paas = strong positive linear relationship
# -1 ke paas = strong negative linear relationship
#  0 ke paas = weak linear relationship
#
# Kyun?
# Samajhne ke liye ki kaun se features similar information
# capture kar sakte hain.
#
# Important:
# Correlation causation prove nahi karta.
# High correlation hone par feature ko automatically delete
# nahi karna.


# STEP 6: MISSING VALUES
# HTML mein: Missing values
#
# Kya check karna hai?
# - Kya kisi column mein missing values hain?
# - Missing values ka chart ya matrix kya dikhata hai?
# - Kya missing values kisi particular group mein hain?
#
# Kyun?
# Missing data model ki training aur predictions ko affect
# kar sakta hai.
#
# Observation:
# Mere report mein missing cells = 0.
# Isliye filhaal missing-value imputation ki zaroorat nahi
# dikh rahi.


# STEP 7: SAMPLE
# HTML mein: Sample
#
# Kya check karna hai?
# - First aur last rows dekho.
# - Values ka format sahi lag raha hai?
# - Koi unexpected value ya repeated label dikh raha hai?
# - Kya koi ID column hai jise samajhna zaroori hai?
#
# Kyun?
# Kuch actual rows dekhkar data ko manually samajhne ke liye.
#
# Important:
# Sample mein sab sahi dikhne se poore dataset ki
# correctness prove nahi hoti.


# ============================================================
# FINAL EDA RULE
#
# 1. Observe   -> Report mein kya dikh raha hai?
# 2. Question  -> Mujhe kya samajhna hai?
# 3. Hypothesis -> Mujhe kya lagta hai aur kyun?
# 4. Test      -> Evidence se apni baat check karo.
# 5. Decision  -> Kya change karna zaroori hai?
#
# Abhi goal model banana nahi hai.
# Pehle data ko samajhna hai, phir preprocessing aur model
# training ki taraf badhna hai.
# ============================================================





# STEP 3: Separate X and y




import pandas as pd

df = pd.read_csv("breast_cancer.csv")


# Target: what the model must predict
y = df["diagnosis"]

# Features: remove the target from the input data
X = df.drop(columns=["diagnosis"])

# Check the result
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 feature rows:")
print(X.head())

print("\nFirst 5 target labels:")
print(y.head())







print("Target column values:")
print(df["target"].value_counts(dropna=False))

print("\nDiagnosis values:")
print(df["diagnosis"].value_counts(dropna=False))

print("\nUnique target values:")
print(df["target"].unique())

print("\nDiagnosis vs target cross-check:")
print(pd.crosstab(df["diagnosis"], df["target"]))



# Actual answer we want to predict
y = df["diagnosis"]

# Input features: remove both versions of the answer
X = df.drop(columns=["diagnosis", "target"])

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Diagnosis in X:", "diagnosis" in X.columns)
print("Target in X:", "target" in X.columns)


# Step 4 — Baseline prediction

# Step 4.2 — Majority-class baseline implement kar


# STEP 4: Majority-class baseline

# Count each diagnosis
class_counts = y.value_counts()
print("Class counts:")
print(class_counts)

# Find the most common diagnosis
majority_class = class_counts.idxmax()
print("\nMajority class:", majority_class)

# Predict the majority class for every row
baseline_predictions = [majority_class] * len(y)

# Measure baseline accuracy
correct = sum(
    actual == predicted
    for actual, predicted in zip(y, baseline_predictions)
)

baseline_accuracy = correct / len(y)

print("Correct predictions:", correct)
print("Total predictions:", len(y))
print("Baseline accuracy:", round(baseline_accuracy * 100, 2), "%")



# Step 5 — Train-test split


# STEP 5: Train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())




# Step 6 — Logistic Regression se pehle Feature Scaling


# STEP 6.1: Check feature scales

print("First 5 training rows:")
print(X_train.head())

print("\nFeature ranges:")
print((X_train.max() - X_train.min()).sort_values(ascending=False).head())

# Step 6.2 — Pehle ek important rule



# STEP 6.2: Feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original training shape:", X_train.shape)
print("Scaled training shape:", X_train_scaled.shape)

print("Original test shape:", X_test.shape)
print("Scaled test shape:", X_test_scaled.shape)





# STEP 6.3: Train Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("First 10 actual diagnoses:")
print(y_test.head(10).to_list())

print("\nFirst 10 predicted diagnoses:")
print(y_pred[:10])



# STEP 7.1: Evaluate model accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Step 7.2 — Baseline vs Model


# STEP 7.3: Confusion matrix

from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["benign", "malignant"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))




# STEP 8.1: Find misclassified test cases

results = X_test.copy()

results["actual"] = y_test
results["predicted"] = y_pred

errors = results[results["actual"] != results["predicted"]]

print("Total incorrect predictions:", len(errors))
print("\nMisclassified cases:")
print(errors[["actual", "predicted"]])



# STEP 8.2: Inspect misclassified cases

print("Misclassified case features:")

print(
    errors[
        [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "actual",
            "predicted"
        ]
    ].round(3)
)



# STEP 8.3: Compare correct vs incorrect predictions

results["correct"] = results["actual"] == results["predicted"]

print("Correct predictions:")
print(
    results.loc[results["correct"], ["mean radius", "mean texture"]]
    .describe()
    .round(2)
)

print("\nIncorrect predictions:")
print(
    results.loc[~results["correct"], ["mean radius", "mean texture"]]
    .describe()
    .round(2)
)




# STEP 8.6: Inspect malignant probabilities

print("Model classes:", model.classes_)

probabilities = model.predict_proba(X_test_scaled)

malignant_index = list(model.classes_).index("malignant")

malignant_probabilities = probabilities[:, malignant_index]

print("\nFirst 10 malignant probabilities:")
print(malignant_probabilities[:10].round(3))




# STEP 8.7: Compare classification thresholds

from sklearn.metrics import recall_score, confusion_matrix

for threshold in [0.5, 0.4, 0.3, 0.2]:
    threshold_predictions = [
        "malignant" if probability >= threshold else "benign"
        for probability in malignant_probabilities
    ]

    malignant_recall = recall_score(
        y_test,
        threshold_predictions,
        pos_label="malignant"
    )

    cm = confusion_matrix(
        y_test,
        threshold_predictions,
        labels=["benign", "malignant"]
    )

    print(f"\nThreshold: {threshold}")
    print(f"Malignant Recall: {malignant_recall:.2%}")
    print("Confusion Matrix:")
    print(cm)




from sklearn.metrics import precision_score

malignant_precision = precision_score(
    y_test,
    threshold_predictions,
    pos_label="malignant"
)

print(f"Malignant Precision: {malignant_precision:.2%}")




# STEP 9.1: Cross-validation setup

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Cross-validation setup ready!")
print("Training samples:", len(X_train))
print("Number of folds:", cv.get_n_splits())




# STEP 9.2: Training data par out-of-fold probabilities

from sklearn.model_selection import cross_val_predict
import numpy as np

# Har training row ki probability us model se milegi
# jisne us row ko training ke liye use nahi kiya.
oof_probabilities = cross_val_predict(
    cv_pipeline,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)

# Model classes mein malignant ka column find karo
malignant_index = list(sorted(y_train.unique())).index("malignant")
# Sirf malignant hone ki probabilities
oof_malignant_probabilities = oof_probabilities[:, malignant_index]

print("OOF probabilities shape:", oof_probabilities.shape)
print("Malignant probabilities shape:", oof_malignant_probabilities.shape)
print("First 5 probabilities:", oof_malignant_probabilities[:5])



# STEP 9.3: Training data par threshold comparison

from sklearn.metrics import recall_score, confusion_matrix

for threshold in [0.5, 0.4, 0.3, 0.2]:
    oof_pred = np.where(
        oof_malignant_probabilities >= threshold,
        "malignant",
        "benign"
    )

    recall = recall_score(
        y_train,
        oof_pred,
        pos_label="malignant"
    )

    cm = confusion_matrix(
        y_train,
        oof_pred,
        labels=["benign", "malignant"]
    )

    print(f"\nThreshold: {threshold}")
    print(f"Malignant Recall: {recall:.2%}")
    print("Confusion Matrix:")
    print(cm)



# STEP 9.5: Final test evaluation with threshold 0.3

from sklearn.metrics import confusion_matrix, classification_report

# Training data par fit hua original model use karo
test_probabilities = model.predict_proba(X_test_scaled)

malignant_index = list(model.classes_).index("malignant")
test_malignant_probabilities = test_probabilities[:, malignant_index]

threshold = 0.3

y_test_pred = np.where(
    test_malignant_probabilities >= threshold,
    "malignant",
    "benign"
)

print("Threshold:", threshold)
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_test_pred,
    labels=["benign", "malignant"]
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    labels=["benign", "malignant"],
    zero_division=0
))
