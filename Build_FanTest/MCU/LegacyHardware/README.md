# Legacy Hardware

This folder keeps the older MCU builds separate from `MCU/newHardware`.

## Fan Test Sketch

Use [legacyFanStream/legacyFanStream.ino](legacyFanStream/legacyFanStream.ino)
for the controlled one-MPU fan test.

This sketch mirrors the current `newHardware` host handshake:

- baud: `1000000`
- firmware prints:
  `Sample struct size (bytes): 16`
- host sends: `START\n`
- packet format: `<I6h`
- packet fields:
  `t_us ax ay az gx gy gz`
- buffered binary stream:
  `32` samples per send block

## Legacy Pins

- `SDA_PIN = 4`
- `SCL_PIN = 5`
- MPU address: `0x68`

## Notes

- `MCU/newHardware` is left alone for the hardware team workflow.
- This legacy fan sketch is the isolated one-sensor adaptation path.
- The active Python `src/` pipeline is expected to match this one-MPU packet.
