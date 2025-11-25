# %% [markdown]
# 03 - modelo con preprocesado y SVM

# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

# %%
train_path = 'data/train_preprocessed.csv'
test_path = 'data/test_preprocessed.csv'

if not os.path.exists(train_path):
    train_path = 'train.csv'

train = pd.read_csv(train_path)

TARGET_COL = 'target'
if TARGET_COL not in train.columns:
    possible = [c for c in train.columns if c.lower() in ('target','label','y')]
    TARGET_COL = possible[0]

X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# %%
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=RANDOM_STATE))
])

param_grid = {
    'svc__kernel': ['rbf', 'linear'],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

# %%
grid.fit(X_train, y_train)
best = grid.best_estimator_

val_pred = best.predict(X_val)
print('Accuracy:', accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

# %%
os.makedirs('models', exist_ok=True)
import joblib
joblib.dump(best, 'models/svm_model.joblib')

# %%
if os.path.exists(test_path):
    test = pd.read_csv(test_path)
    id_col = None
    for c in ['id','ID','Id']:
        if c in test.columns:
            id_col = c
            break
    if id_col is None:
        ids = np.arange(len(test))
    else:
        ids = test[id_col]

    X_test = test.drop(columns=[col for col in test.columns if col in ['id','ID','Id']], errors='ignore')
    preds = best.predict_proba(X_test)[:,1]

    sub = pd.DataFrame({'id': ids, 'target': preds})
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/submission_svm.csv', index=False)

# %%
print("Notebook 03 ejecutado correctamente.")
