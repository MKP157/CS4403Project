import cupy as cp
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# Encode text labels
label_encoder = LabelEncoder()


df_2018 = pd.read_parquet("./data/clean_joined_2018.parquet").dropna()
df_2018 = df_2018[:1_000_000]
print(df_2018)


df_2018["MFR"] = label_encoder.fit_transform(df_2018["MFR"])


y_labelled = []
for i, row in df_2018.iterrows():
    if row['Cancelled']:
        y_labelled.append('cancelled')
    elif row['Diverted']:
        y_labelled.append('diverted')
    elif row['ArrDelayMinutes'] > 0:
        y_labelled.append('delayed')
    else:
        y_labelled.append('ok')
        
y = label_encoder.fit_transform(y_labelled)

X = df_2018.drop(columns=["Tail_Number", "N-NUMBER", 'Cancelled', 'Diverted'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    np.array(y),
    test_size=0.2,
    random_state=42
)

# Train XGBoost on GPU
xgb_model = xgb.XGBClassifier(
    num_class=len(label_encoder.classes_),  # Automatically set class count
    tree_method="hist",
    device='cuda',
    objective="multi:softprob",
    n_estimators=100,
    eval_metric='mlogloss',  # Recommended for multi-class [1][7]
	enable_categorical=False,
    learning_rate=0.5,
)

xgb_model.fit(X_train, y_train)

X_cp = cp.array(X_test)
preds = xgb_model.predict(X_cp)
print(preds)
#print(classification_report(y_test, preds))

# Feature importance
#xgb.plot_importance(xgb_model)