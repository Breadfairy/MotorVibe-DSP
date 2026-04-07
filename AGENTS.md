# AGENTS.md

## Scope
These rules apply to the entire repository.

## Project Goal
Build simple scripts for motor vibration analysis using a combination of dsp 
and ML. there should be a live mode.

## Coding Style
- Keep code extremely minimal and purely direct signal and data flow
- seperate each .py file with lines of `###` and a section heading  in the following order:
  - Imports
  - variables/constants
  - helpers
  - classes or main functions
- at all costs, do not add helpers or classes unless absolutely necessary for the core logic.
- each .py file should be a top down script with a clear flow of data and logic.
- No error handling, fallback code or any other handling that gets it the way fo the core logic.
- i will add the handling as it arrises or on suggestions only
- no CLI, the .py files will be completely self contained for 1 function. there can be minimal argv parsing for file patsh and names. but keep most configurables at the top of the file as a variable or constant.
- each .py file will be build around the necessary fucntionality. for example. training.py is purely for ML training. live.py is purly for live sensing. capture.py is purel for data capture.
- obviously each of the seperated .py files mentioned above will need to use shared mathematics and functions to reduce code duplications such as signal processing and charting. but these should be in a seperate .py file that is imported as a module and not a helper function within the main .py files.
- limit line length to 80 characters
- No globals for signal arrays; return data from functions.
- Dictionaries are allowed for raw and derived signal containers.
- Prefer explicit sensor and signal names as dictionary keys.
- Do not add try/except blocks unless explicitly requested.
- Keep descriptor comments at the top of each function.
- Do not assign default values in function arguments when those values are
  defined outside and passed in.

from this point forwar the readme maybe need to be updated based on the rules above.


## Data Flow Rules
basic flow of data is as follows:
  read data > buffer data > build signals > ML and rules > visualisation.

Current target module flow:
- `buffer.py`
  - build fixed-size buffers from CSV or serial rows
- `signals.py`
  - build time-domain and frequency-domain signals from the current buffer
- `ml.py`
  - train and run the PyTorch model
- `rules.py`
  - run non-ML logic and health thresholds
- `charting.py`
  - display or save charts
- `main.py`
  - call the stages sequentially in a clear top-down script flow

For `signals.py`, prefer this structure unless asked otherwise:
- Required function names:
  - `readCSV`
  - `liveSense`
  - `buildSignals`
- `readCSV` uses pandas to read CSV rows for buffer and signal building.
- `liveSense` reads serial rows for buffer and signal building.
- `buildSignals` converts columns directly to NumPy `float64` arrays and
  returns named raw and derived signals.
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
