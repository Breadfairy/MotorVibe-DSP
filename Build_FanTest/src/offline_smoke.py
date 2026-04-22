################################################################################
# Imports                                                                      #
################################################################################
import math
import struct

import pandas as pd

import buffer
import ml
import signals

################################################################################
# main functions                                                               #
################################################################################


for mode in ["sixaxis_1k", "gyro_8k"]:
    fmt = buffer.config(mode)["sampleFmt"]
    rows = []
    n = int(buffer.sampleRate(mode))
    for i in range(n):
        t = i / buffer.sampleRate(mode)
        if mode == "sixaxis_1k":
            vals = [
                int(600 * math.sin(t * 20.0)),
                int(500 * math.sin(t * 23.0)),
                int(400 * math.sin(t * 17.0)),
                int(300 * math.sin(t * 60.0)),
                int(260 * math.sin(t * 63.0)),
                int(220 * math.sin(t * 57.0)),
            ]
            payload = struct.pack(fmt, *vals)
        if mode == "gyro_8k":
            vals = [
                int(320 * math.sin(t * 300.0)),
                int(280 * math.sin(t * 340.0)),
                int(240 * math.sin(t * 380.0)),
            ]
            payload = struct.pack(fmt, *vals)
        head = {
            "mode": mode,
            "startSample": i,
            "sampleRateHz": int(buffer.sampleRate(mode)),
        }
        rows += buffer.decodeRows(head, payload)
    dataFrame = pd.DataFrame(rows, columns=buffer.csvCols(mode))
    sig = signals.buildSignals(dataFrame, mode)
    vec = ml.modelVector(sig)
    net = ml.model(mode)
    prob = ml.runModel(net, ml.featureTensor(vec[None, :]))
    print("mode:", mode)
    print("rows:", len(rows))
    print("vectorShape:", vec.shape)
    print("fundHz:", sig["fundHz"])
    print("topLabel:", ml.topLabel(prob))
    print("offlineSmoke: PASS")
