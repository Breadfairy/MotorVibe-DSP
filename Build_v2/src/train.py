# Trains the fault classifiers from labelled CSV recordings.
from pathlib import Path
import os

buildDir = Path(__file__).resolve().parents[1]
mplConfigDir = Path(os.environ.get("TMPDIR", "/tmp")) / "Build_FanTest_v2_matplotlib"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mplConfigDir))
os.environ.setdefault("XDG_CACHE_HOME", str(mplConfigDir))

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import data
import config
import signals

# Defines the project paths used by the training pipeline.
trainDir = buildDir / "data"
modelDir = buildDir / "outputs" / "models"
chartDir = buildDir / "outputs" / "ML-charts"
reportDir = buildDir / "outputs"

# Defines the windowing and reproducibility settings used for all models.
classLabels = config.classLabels
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 0.25
splitSeed = 42
plotLabels = [
    "Off",
    "Healthy",
    "8.5v Drop",
    "8.0v Drop",
    "7.5v Drop",
    "Obstruction",
    "Imbalance",
]
plotTitles = {
    "logistic_regression": "Logistic Regression",
    "logistic_regression_C0p1": "Weighted Logistic Regression",
    "random_forest": "Random Forest",
    "rbf_svc": "Support Vector Classification with RBF",
}
summaryNames = {
    "logistic_regression": "Logistic Regression",
    "logistic_regression_C0p1": "Weighted Logistic Regression",
    "random_forest": "Random Forest",
    "rbf_svc": "Support Vector Classification",
}
healthLabels = ["good", "obstruction", "imbalance"]
healthReportPath = reportDir / "final_health_report.txt"


# Returns the saved model path for a catalogued model name.
def catalogPath(modelName):
    for _, name, fileName in config.modelCatalog:
        if name == modelName:
            return modelDir / fileName
    raise KeyError(f"unknown model name: {modelName}")


# Finds correctly named CSV recordings for one class folder.
def sourceCsvs(classLabel):
    classDir = trainDir / classLabel
    if not classDir.exists():
        raise SystemExit(f"missing class folder: {classDir}")

    csvPaths = sorted(classDir.glob("*.csv"))
    for csvPath in csvPaths:
        if not (csvPath.name.startswith("train_") or csvPath.name.startswith("val_")):
            raise SystemExit(
                f"CSV must start with train_ or val_: {csvPath}")
        if not hasEnoughRows(csvPath):
            raise SystemExit(f"CSV is too short for a training window: {csvPath}")
    return csvPaths


# Splits one class folder by explicit train_/val_ filename prefixes.
def plannedCsvSplit(label, csvPaths):
    trainCsvs = [path for path in csvPaths if path.name.startswith("train_")]
    valCsvs = [path for path in csvPaths if path.name.startswith("val_")]
    if len(trainCsvs) == 0:
        raise SystemExit(f"no train_ CSV files for class: {label}")
    return trainCsvs, valCsvs


# Returns true when the CSV can produce at least one training window.
def hasEnoughRows(csvPath):
    minLines = int(sampleRate * winSecs) + 1
    with csvPath.open("r", newline="") as csvFile:
        for lineIndex, _ in enumerate(csvFile, start=1):
            if lineIndex >= minLines:
                return True
    return False


# Converts CSV recordings into labelled feature rows.
def addCsvFeatures(csvPaths, classIndex, featureRows, classRows):
    for csvPath in csvPaths:
        df = data.readCsv(csvPath)
        df = data.cleanFrame(df)
        windows = data.windowFrames(df, sampleRate, winSecs, stepSecs)
        for windowFrame in windows:
            timeData = signals.timeSignals(windowFrame, sampleRate)
            freqData = signals.fftSignals(
                timeData,
                sampleRate,
                signals.fftConfig,
            )
            featureRows.append(signals.modelInput(timeData, freqData))
            classRows.append(classIndex)


# Defines every candidate model trained by this script.
def modelCandidates():
    return [
        (
            "logistic_regression",
            catalogPath("logistic_regression"),
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ),
        (
            "logistic_regression_C0p1",
            catalogPath("logistic_regression_C0p1"),
            "logistic_regression_C0p1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=0.1,
                ),
            ),
        ),
        (
            "random_forest",
            catalogPath("random_forest"),
            "random_forest",
            RandomForestClassifier(
                n_estimators=250,
                random_state=splitSeed,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
        (
            "rbf_svc",
            catalogPath("rbf_svc"),
            "rbf_svc",
            make_pipeline(
                StandardScaler(),
                SVC(
                    C=3.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=splitSeed,
                ),
            ),
        ),
    ]


# Defines the three-state health models trained after the full classifiers.
def healthModelCandidates():
    return [
        (
            "health_logistic_regression",
            modelDir / "health_logistic_regression.joblib",
            "health_logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ),
        (
            "health_logistic_regression_C0p1",
            modelDir / "health_logistic_regression_C0p1.joblib",
            "health_logistic_regression_C0p1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=0.1,
                ),
            ),
        ),
        (
            "health_random_forest",
            modelDir / "health_random_forest.joblib",
            "health_random_forest",
            RandomForestClassifier(
                n_estimators=250,
                random_state=splitSeed,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
        (
            "health_rbf_svc",
            modelDir / "health_rbf_svc.joblib",
            "health_rbf_svc",
            make_pipeline(
                StandardScaler(),
                SVC(
                    C=3.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=splitSeed,
                ),
            ),
        ),
    ]


# Saves one validation confusion matrix as a report-ready PNG chart.
def saveConfusionPlot(matrix, labels, title, plotPath):
    rowTotals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=rowTotals != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(
        percent,
        cmap="Blues",
        norm=PowerNorm(gamma=0.28, vmin=0.0, vmax=100.0),
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if percent[row, col] >= 50.0 else "#222222"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{percent[row, col]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("row percentage")
    fig.tight_layout()

    fig.savefig(plotPath, dpi=180)
    plt.close(fig)
    return plotPath


# Builds the output PNG path for one model confusion matrix.
def confusionPlotPath(modelName):
    return chartDir / f"confusion_matrix_{modelName}.png"


# Builds the output PNG path for one binary health confusion matrix.
def healthConfusionPlotPath(modelName):
    return chartDir / f"health_confusion_matrix_{modelName}.png"


# Builds the persisted bundle consumed by live.py.
def modelBundle(model, labels, modelMode, trainAcc, valAcc):
    return {
        "model": model,
        "labelNames": labels,
        "sampleRate": sampleRate,
        "winSecs": winSecs,
        "stepSecs": stepSecs,
        "featureNames": signals.featureNames(),
        "featureMode": "motor_compact_low_order_bearing_orders",
        "modelMode": modelMode,
        "trainAcc": trainAcc,
        "valAcc": valAcc,
    }


# Builds source-separated training and validation feature matrices.
def buildFeatureSets():
    featureRows = []
    classRows = []
    valFeatureRows = []
    valClassRows = []

    labels = list(classLabels)
    for classIndex, label in enumerate(labels):
        trainCsvs, valCsvs = plannedCsvSplit(label, sourceCsvs(label))
        addCsvFeatures(trainCsvs, classIndex, featureRows, classRows)
        addCsvFeatures(valCsvs, classIndex, valFeatureRows, valClassRows)

    featureMatrix = np.vstack(featureRows).astype(np.float32)
    classVector = np.array(classRows, dtype=np.int64)
    valFeatureMatrix = np.vstack(valFeatureRows).astype(np.float32)
    valClassVector = np.array(valClassRows, dtype=np.int64)
    return labels, featureMatrix, classVector, valFeatureMatrix, valClassVector


# Builds the final good/obstruction/imbalance health feature matrices.
def buildHealthFeatureSets():
    featureRows = []
    classRows = []
    valFeatureRows = []
    valClassRows = []

    trainCsvs, valCsvs = plannedCsvSplit("good", sourceCsvs("good"))
    addCsvFeatures(trainCsvs, 0, featureRows, classRows)
    addCsvFeatures(valCsvs, 0, valFeatureRows, valClassRows)

    for classIndex, label in enumerate(["obstruction", "imbalance"], start=1):
        trainCsvs, valCsvs = plannedCsvSplit(label, sourceCsvs(label))
        addCsvFeatures(trainCsvs, classIndex, featureRows, classRows)
        addCsvFeatures(valCsvs, classIndex, valFeatureRows, valClassRows)

    featureMatrix = np.vstack(featureRows).astype(np.float32)
    classVector = np.array(classRows, dtype=np.int64)
    valFeatureMatrix = np.vstack(valFeatureRows).astype(np.float32)
    valClassVector = np.array(valClassRows, dtype=np.int64)
    return featureMatrix, classVector, valFeatureMatrix, valClassVector


# Trains all final good/obstruction/imbalance health models and writes the report.
def trainHealthApplicationModels():
    featureMatrix, classVector, valFeatureMatrix, valClassVector = (
        buildHealthFeatureSets()
    )
    summaryRows = []
    reports = []

    for modelName, savePath, modelMode, model in healthModelCandidates():
        model.fit(featureMatrix, classVector)
        trainPred = model.predict(featureMatrix)
        valPred = model.predict(valFeatureMatrix)

        trainAcc = accuracy_score(classVector, trainPred)
        valAcc = accuracy_score(valClassVector, valPred)
        labelIndexes = list(range(len(healthLabels)))
        valMatrix = confusion_matrix(
            valClassVector,
            valPred,
            labels=labelIndexes,
        )
        report = classification_report(
            valClassVector,
            valPred,
            labels=labelIndexes,
            target_names=healthLabels,
            zero_division=0,
        )

        chartDir.mkdir(parents=True, exist_ok=True)
        saveConfusionPlot(
            valMatrix,
            healthLabels,
            plotTitles[modelName.replace("health_", "")],
            healthConfusionPlotPath(modelName),
        )

        savePath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            modelBundle(model, healthLabels, modelMode, trainAcc, valAcc),
            savePath,
        )

        summaryName = summaryNames[modelName.replace("health_", "")]
        summaryRows.append((summaryName, valAcc))
        reports.append((summaryName, trainAcc, valAcc, valMatrix, report))

        print()
        print(modelName)
        print("valAcc:", f"{valAcc * 100.0:.2f}%")
        print("valConfusion:")
        print(valMatrix)
        print("valReport:")
        print(report)

    print()
    for displayName, valAcc in summaryRows:
        print(f"{displayName:<34} Accuracy: {valAcc * 100.0:.2f}%")

    reportDir.mkdir(parents=True, exist_ok=True)
    with healthReportPath.open("w") as reportFile:
        reportFile.write("Health Classification Report\n")
        reportFile.write("Training classes: good, obstruction, imbalance\n")
        reportFile.write("Excluded classes: off, voltSag8p5, voltSag8p0, voltSag7p5\n\n")
        for displayName, trainAcc, valAcc, valMatrix, report in reports:
            reportFile.write(f"{displayName}\n")
            reportFile.write(f"trainAcc: {trainAcc * 100.0:.2f}%\n")
            reportFile.write(f"valAcc: {valAcc * 100.0:.2f}%\n")
            reportFile.write("valConfusion:\n")
            reportFile.write(f"{valMatrix}\n")
            reportFile.write("valReport:\n")
            reportFile.write(report)
            reportFile.write("\n\n")

    print("healthReport:", healthReportPath)
    return summaryRows


# Trains every configured sklearn model and saves the model bundles.
def main():
    labels, featureMatrix, classVector, valFeatureMatrix, valClassVector = (
        buildFeatureSets()
    )
    labelIndexes = list(range(len(labels)))
    summaryRows = []

    # Trains and evaluates each configured model on the same feature split.
    for modelName, savePath, modelMode, model in modelCandidates():
        model.fit(featureMatrix, classVector)
        trainPred = model.predict(featureMatrix)
        valPred = model.predict(valFeatureMatrix)

        trainAcc = accuracy_score(classVector, trainPred)
        valAcc = accuracy_score(valClassVector, valPred)
        valMatrix = confusion_matrix(
            valClassVector,
            valPred,
            labels=labelIndexes,
        )
        report = classification_report(
            valClassVector,
            valPred,
            labels=labelIndexes,
            target_names=labels,
            zero_division=0,
        )

        savePath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            modelBundle(model, labels, modelMode, trainAcc, valAcc),
            savePath,
        )

        chartDir.mkdir(parents=True, exist_ok=True)
        saveConfusionPlot(
            valMatrix,
            plotLabels,
            plotTitles[modelName],
            confusionPlotPath(modelName),
        )
        summaryRows.append((summaryNames[modelName], valAcc))

        print()
        print(modelName)
        print("valAcc:", f"{valAcc * 100.0:.2f}%")
        print("valConfusion:")
        print(valMatrix)
        print("valReport:")
        print(report)

    print()
    for displayName, valAcc in summaryRows:
        print(f"{displayName:<34} Accuracy: {valAcc * 100.0:.2f}%")

    trainHealthApplicationModels()

    print()
    print("modelDir:", modelDir)


if __name__ == "__main__":
    main()
