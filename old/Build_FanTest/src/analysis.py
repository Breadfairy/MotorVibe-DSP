################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys

import buffer
import charting
import signals

################################################################################
# variables/constants                                                          #
################################################################################
rootDir = Path(__file__).resolve().parents[1]
csvPath = rootDir / "data" / "capture" / "test1.csv"
outDir = rootDir / "outputs" / "charts"

################################################################################
# main functions                                                               #
################################################################################


if len(sys.argv) > 1:
    csvPath = Path(sys.argv[1])

dataFrame = signals.readCSV(csvPath)
mode = buffer.modeFromCols(dataFrame.columns)
sig = signals.buildSignals(dataFrame, mode)
saveDir = outDir / csvPath.stem
saveDir.mkdir(parents=True, exist_ok=True)
timePath = saveDir / "timeGrid.png"
fftPath = saveDir / "fftGrid.png"
summaryPath = saveDir / "summary.png"
charting.plotTimeGrid(sig, timePath)
charting.plotFftGrid(sig, fftPath)
charting.plotSummary(sig, summaryPath)

print("csvPath:", csvPath)
print("mode:", mode)
print("timePath:", timePath)
print("fftPath:", fftPath)
print("summaryPath:", summaryPath)
