# Motor Signal Pipeline (Python)

## Current Structure
- `main.py` is the orchestration module.
- `signals.py` reads/builds arrays from CSV or live serial data.
- `charting.py` handles plotting and saving charts.
- `analytics.py` is reserved for math operations on arrays.
- `dsp.py` is present but not used in the current `main.py` flow.

## Data Schema
The signal build and plotting flow expects this exact column order:

1. `Amostra`
2. `Acel_X`
3. `Acel_Y`
4. `Acel_Z`
5. `Giro_X`
6. `Giro_Y`
7. `Giro_Z`
8. `Temperatura`

In code, `Amostra` is mapped to array variable `sample`.

## Main Flow
`main.py` supports two modes:

- `csv`: reads a CSV, builds signal arrays, prints quick checks, and plots.
- `live`: reads serial data, builds signal arrays, and prints quick checks.

Mode is passed by argv:

```bash
python3 main.py csv
python3 main.py live
```

## CSV Mode
- CSV path is currently hardcoded in `main.py`:
  - `data/testData/nivel2.csv` (speed 2 dataset)
- The script prints:
  - `csvPath`
  - `csvDataHead`
  - first 5 values of each built array

## Live Mode
- Serial settings are configured in `main.py`:
  - `serialPort`
  - `baudRate`
  - `serialDelimiter`
  - `serialTimeout`
  - `serialEncoding`
- `runLive` currently calls `signals.liveSense(..., sample_count=5000, ...)`.
- The MCU must stream one line per sample in this order:
  - `Amostra,Acel_X,Acel_Y,Acel_Z,Giro_X,Giro_Y,Giro_Z,Temperatura`

## Charting
- Plot function: `charting.plotRaw(...)`
- Output path base: `outputs/csv`
- Output filename pattern:
  - `<csv_stem>_plotRaw.png`
- Current plot limits each chart to the first `100` samples.
- One figure is produced with three subplots:
  - top-left: `Acel_X`, `Acel_Y`, `Acel_Z` vs `sample`
  - top-right: `Giro_X`, `Giro_Y`, `Giro_Z` vs `sample`
  - bottom full width: `Temperatura` vs `sample`

## Dependencies
- `numpy`
- `pandas`
- `pyserial`
- `matplotlib`
