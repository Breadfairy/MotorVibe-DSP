# AGENTS.md

## Scope
These rules apply to the entire repository.

## Project Goal
Build a simple Python signal, dsp and visualisation pipeline for motor sensing:
- serial MCU data -> arrays
- CSV test data -> arrays
- DSP and analytics can consume those arrays

## Coding Style
- Keep code extremely minimal and purely direct signal and data flow.
- No error handling, fallback code or any other handling that gets it the way fo the core logic.
- i will add the handling later.
- Prefer short, explicit functions over abstractions.
- no CLI wrappers unless asked for
- Use clear variable names in camelCase code conventions
- limit line length to 80 characters
- No globals for signal arrays; return data from functions.
- No dictionaries for sensor-array outputs unless explicitly requested.
- Do not add try/except blocks unless explicitly requested.
- Keep descriptor comments at the top of each function.
- Do not assign default values in function arguments when those values are
  defined outside and passed in.

## Data Flow Rules
basic flow of data is as follows:
  read data > build signals > DSP and build dsp arrays > visualisation.

For `signals.py`, keep this exact simple template unless asked otherwise:
- Required function names:
  - `readCSV`
  - `liveSense`
  - `buildSignals`
- `readCSV` uses pandas to read CSV, then calls `buildSignals`.
- `liveSense` reads serial rows, builds a pandas DataFrame, then calls
  `buildSignals`.
- `buildSignals` converts columns directly to NumPy `float64` arrays and
  returns:
  `VibeOneA, VibeTwoA, HeatOne`
- Use fixed sensor columns:
  `["VibeOneA", "VibeTwoA", "HeatOne"]`
- Do not add header/schema validation unless requested.

## Dependencies
- Use `numpy` for arrays.
- Use `pandas` for CSV/table handling.
- Use `pyserial` (`serial`) for live MCU serial input.

## Platform Notes
- Development may happen on macOS.
- Runtime may happen on Windows.
- Avoid hardcoding OS-specific serial port names.

## Testing
- Keep tests lightweight and fast.
- Prefer small smoke tests for data flow first.
- Do not add heavy frameworks unless requested.

## Change Discipline
- Make targeted edits only.
- Do not refactor unrelated files.
- Preserve existing behavior unless the task asks to change it.
