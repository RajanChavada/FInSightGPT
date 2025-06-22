import shap
import pandas as pd



def explain_model(clf, X_train, X_test):
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test[X_train.columns].apply(pd.to_numeric, errors='coerce')

    X_train_clean = X_train.dropna()
    X_test_clean = X_test.dropna()
    explainer = shap.Explainer(clf, X_train_clean)
    shap_values = explainer(X_test_clean)

    shap_values_class1 = shap.Explanation(
        values=shap_values.values[:, :, 1],
        base_values=shap_values.base_values[:, 1],
        data=X_test_clean,
        feature_names=X_test_clean.columns
    )
    shap.plots.bar(shap_values_class1, max_display=15)
    shap.plots.beeswarm(shap_values_class1, max_display=15)
