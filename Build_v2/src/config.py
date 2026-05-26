# Defines shared labels and model filenames used by capture, training, and live.

# Lists the class folders accepted by capture and training.
classLabels = [
    "off",
    "good",
    "voltSag8p5",
    "voltSag8p0",
    "voltSag7p5",
    "obstruction",
    "imbalance",
]

# Maps live model numbers to trained model names and output files.
modelCatalog = [
    ("1", "logistic_regression", "logistic_regression.joblib"),
    ("2", "logistic_regression_C0p1", "logistic_regression_C0p1.joblib"),
    ("3", "random_forest", "random_forest.joblib"),
    ("4", "rbf_svc", "rbf_svc.joblib"),
]
