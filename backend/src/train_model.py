import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def normalize_column(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) if max_val != min_val else pd.Series([0] * len(series), index=series.index)


def train_classifier(df):
    X = df.drop(columns=['made_q3'])
    y = df['made_q3']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    print("Stratified CV Accuracy:", scores.mean())

    return clf, X_test, y_test