# ESP32 Dummy Serial Harness

This is the simple ESP32 path if the board firmware uses Arduino-core
`Serial`.

It sends newline-terminated CSV rows at `115200` baud in the same
17-field order used by the Python live pipeline:

`sample,mpu1AccX,mpu1AccY,mpu1AccZ,mpu1GyrX,mpu1GyrY,mpu1GyrZ,`
`mpu1Temp,mpu2AccX,mpu2AccY,mpu2AccZ,mpu2GyrX,mpu2GyrY,mpu2GyrZ,`
`mpu2Temp,ds18b20One,ds18b20Two`

## What `Serial` Means Here

On ESP32 with Arduino core:

- `Serial.begin(115200)` opens the USB or UART serial link
- `Serial.print(...)` writes text fields
- `Serial.println(...)` ends one CSV row with a newline

This is not the Python `serial` package.

## Build And Flash

This Makefile expects `arduino-cli` and the ESP32 core to already be
installed.

Update the `PORT` and `BOARD_FQBN` values in `Makefile` if your board or
device path differ.

Then run:

```sh
cd serial_harness/esp32_dummy_serial
make compile
make flash
```

## Read On The Computer

Your existing host-side monitor can read it directly at `115200`:

```sh
python serial_harness/serialPrint.py
```

If you want to use it with `src/main.py`, point `serialPort` at the ESP32
USB serial device and keep `baudRate = 115200`.
