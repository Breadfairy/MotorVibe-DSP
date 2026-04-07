# Simplified Capture

based of the example 1khz.ino and data_logging.py file provided.
this is for a single MPU sensor initially.

Open `capture.py` in VS Code or whatever and check the settings near
the top of the file:

- `captureTime` (seconds)
- `fileName`    (the name of the csv label for ML data)
- `serialPort`  (use `ls /dev/tty*` on Linux or check device manager on Windows)
- `baudRate`

Once the firmware is loaded onto the board and the board is connected to the
computer, run the command from the terminal:

python3 capture.py

That will save a CSV file into the `ML_data` folder using the default
`fileName` written in the script.

If you want to quickly choose a different file name from the terminal, you can
run:

python3 capture.py test1

That will save a csv called test1.csv into: ML_data/test1.csv

the ML_Data folder is created by the script. it checks for its existance and
makes it if needed.

You do not need to type `.csv` at the end. The script adds that for you.

When it starts, it prints a basic message saying the capture has started.
When it finishes, it prints that the capture is complete, how many rows were
logged, and where the file was saved.

## Expected Problems From Chatty.

- The Python and firmware formats might stop matching.
  If the `.ino` file changes the data order, number of values, or value types,
  then `capture.py` also needs to change or the CSV will be wrong.

- A second MPU6050 can cause an I2C address clash.
  Usually one MPU needs one address and the other MPU needs the other address.
  If both try to use the same address, they will not both work properly.

- A different dev board can change pins and serial behaviour.
  The new board might use different I2C pins, different USB behaviour, or act
  slightly differently at the same baud rate.

- DS18B20 is much slower than the MPU6050.
  The IMU can run fast. The temperature sensor usually cannot be read the same
  way at the same speed, so it may need slower updates.

- More sensors means more serial data.
  As more values are added to each record, the serial link has to move more
  bytes. If the data rate gets too high, logging can become unstable.

- Raw binary has no safety markers right now.
  This simple setup is easy to read, but if bytes get shifted or lost, the
  data can become misaligned and harder to debug.

- The unpack format in Python is easy to forget.
  If the firmware changes from one record layout to another, `recordFormat`
  in `capture.py` must also be updated to match it exactly.

- Timestamp meaning can change without people noticing.
  Make sure everyone agrees on what `t_us` means. It should not be vague.
  It needs to be clearly understood by both the firmware side and Python side.


