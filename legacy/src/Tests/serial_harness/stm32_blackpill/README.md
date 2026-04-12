# STM32 Black Pill Arduino Dummy Serial

This is an Arduino-core build for a Black Pill style STM32 board.

It sends newline-terminated CSV rows at `115200` baud in the same
17-field order used by the Python live pipeline:

`sample,mpu1AccX,mpu1AccY,mpu1AccZ,mpu1GyrX,mpu1GyrY,mpu1GyrZ,`
`mpu1Temp,mpu2AccX,mpu2AccY,mpu2AccZ,mpu2GyrX,mpu2GyrY,mpu2GyrZ,`
`mpu2Temp,ds18b20One,ds18b20Two`

## Board Assumption

Default `Makefile` target:

- `stm32:stm32:GenF4:pnum=BLACKPILL_F411CE`

If your board is a `BlackPill F401CC`, change `BOARD_FQBN` in the
`Makefile`.

## What `Serial` Means On STM32 Arduino

This sketch uses plain `Serial.begin(115200)`.

On STM32 Arduino, `Serial` can point to:

- native USB CDC, if USB CDC support is enabled for the board config
- a hardware UART, if USB CDC is not enabled

So there are two valid ways to read this sketch on your computer:

- direct USB-C if your board config maps `Serial` to USB CDC
- USB-UART adapter if your board config maps `Serial` to a UART

## Build And Flash

This Makefile expects:

- `arduino-cli`
- the STM32 core from STM32duino already installed

Then run:

```sh
cd serial_harness/stm32_blackpill
make compile
make flash
```

Update `PORT` in the `Makefile` before flashing.

## Read On The Computer

Your existing host-side monitor can read the output directly at `115200`:

```sh
python serial_harness/serialPrint.py
```

If you want to use it with `src/main.py`, point `serialPort` at the Black
Pill serial device and keep `baudRate = 115200`.

## Notes

This Makefile keeps the board string minimal on purpose.

If you want a stricter setup for one exact upload path, I can lock it to
one of these:

- native USB DFU upload
- ST-Link SWD upload
- UART bootloader upload
