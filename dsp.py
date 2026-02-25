import numpy as np


# Minimal DSP stage that transforms raw arrays into DSP-ready arrays.
def runDSP(VibeOneA, VibeTwoA, HeatOne):
    dspVibeOneA = np.abs(VibeOneA)
    dspVibeTwoA = np.abs(VibeTwoA)
    dspHeatOne = HeatOne
    return dspVibeOneA, dspVibeTwoA, dspHeatOne
