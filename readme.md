# Motor Signal Pipeline (Python)

This project is a simple motor-signal pipeline. It reads sensor data, builds
raw signal arrays, applies lightweight analytics transforms, and saves charts.

For CSV testing, the project uses an online dataset captured from the same
sensor family as the target hardware. The goal is to keep development close to
real sensor behavior while the live serial path is being tuned.

The signal flow is direct. `main.py` orchestrates the process. `signals.py`
reads the input and builds `rawSignals` as an array of arrays. `analytics.py`
takes `rawSignals` and builds `dspSignals`, including the filtered temperature
trend. `charting.py` takes `dspSignals` and writes the output plot image.

In CSV mode, the current test file is selected in `main.py` and the chart is
saved to `outputs/csv`. In live mode, serial data is read through the same
signal path so both modes stay aligned.

Dependencies are `numpy`, `pandas`, `pyserial`, and `matplotlib`.

```bash
python3 -m pip install numpy pandas pyserial matplotlib
```

Run commands:

```bash
python3 main.py csv
python3 main.py live
```
