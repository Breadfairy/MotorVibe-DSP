#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22

#define MPU1_ADDR 0x68
#define MPU2_ADDR 0x69

#define REG_SMPLRT_DIV    0x19
#define REG_CONFIG        0x1A
#define REG_GYRO_CONFIG   0x1B
#define REG_ACCEL_CONFIG  0x1C
#define REG_INT_ENABLE    0x38
#define REG_ACCEL_XOUT_H  0x3B
#define REG_SIGNAL_RESET  0x68
#define REG_USER_CTRL     0x6A
#define REG_PWR_MGMT_1    0x6B
#define REG_PWR_MGMT_2    0x6C
#define REG_WHO_AM_I      0x75

bool readBytes(uint8_t addr, uint8_t reg, uint8_t *dst, uint8_t count) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(addr, count) != count) {
    return false;
  }

  for (uint8_t i = 0; i < count; i++) {
    dst[i] = Wire.read();
  }
  return true;
}

bool readReg(uint8_t addr, uint8_t reg, uint8_t &value) {
  return readBytes(addr, reg, &value, 1);
}

bool writeReg(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

int16_t toInt16(uint8_t hi, uint8_t lo) {
  return (int16_t)((hi << 8) | lo);
}

void printByte(uint8_t value) {
  if (value < 16) {
    Serial.print("0");
  }
  Serial.print(value, HEX);
}

void printReg(uint8_t addr, const char *name, uint8_t reg) {
  uint8_t value = 0;
  Serial.print("  ");
  Serial.print(name);
  Serial.print(" 0x");
  printByte(reg);
  Serial.print(" = ");
  if (readReg(addr, reg, value)) {
    Serial.print("0x");
    printByte(value);
    Serial.print(" ");
    Serial.println(value);
  } else {
    Serial.println("read_fail");
  }
}

void setupMpuOriginal(uint8_t addr) {
  writeReg(addr, REG_PWR_MGMT_1, 0x01);
  delay(100);
  writeReg(addr, REG_CONFIG, 0x03);
  writeReg(addr, REG_SMPLRT_DIV, 0x00);
  writeReg(addr, REG_GYRO_CONFIG, 0x00);
  writeReg(addr, REG_ACCEL_CONFIG, 0x00);
  writeReg(addr, REG_INT_ENABLE, 0x01);
}

void setupMpuReset(uint8_t addr) {
  writeReg(addr, REG_PWR_MGMT_1, 0x80);
  delay(100);
  writeReg(addr, REG_SIGNAL_RESET, 0x07);
  delay(20);
  writeReg(addr, REG_PWR_MGMT_1, 0x01);
  writeReg(addr, REG_PWR_MGMT_2, 0x00);
  writeReg(addr, REG_USER_CTRL, 0x00);
  writeReg(addr, REG_CONFIG, 0x03);
  writeReg(addr, REG_SMPLRT_DIV, 0x00);
  writeReg(addr, REG_GYRO_CONFIG, 0x00);
  writeReg(addr, REG_ACCEL_CONFIG, 0x00);
  writeReg(addr, REG_INT_ENABLE, 0x01);
}

void printRawSample(uint8_t addr) {
  uint8_t raw[14];
  if (!readBytes(addr, REG_ACCEL_XOUT_H, raw, 14)) {
    Serial.println("  raw read_fail");
    return;
  }

  Serial.print("  raw:");
  for (uint8_t i = 0; i < 14; i++) {
    Serial.print(" ");
    printByte(raw[i]);
  }
  Serial.println();

  Serial.print("  accel:");
  Serial.print(toInt16(raw[0], raw[1]));
  Serial.print(" ");
  Serial.print(toInt16(raw[2], raw[3]));
  Serial.print(" ");
  Serial.print(toInt16(raw[4], raw[5]));
  Serial.print(" gyro:");
  Serial.print(toInt16(raw[8], raw[9]));
  Serial.print(" ");
  Serial.print(toInt16(raw[10], raw[11]));
  Serial.print(" ");
  Serial.println(toInt16(raw[12], raw[13]));
}

void printStatus(uint8_t addr, const char *label) {
  Serial.print(label);
  Serial.print(" addr=0x");
  printByte(addr);
  Serial.println();

  printReg(addr, "WHO_AM_I", REG_WHO_AM_I);
  printReg(addr, "PWR_MGMT_1", REG_PWR_MGMT_1);
  printReg(addr, "PWR_MGMT_2", REG_PWR_MGMT_2);
  printReg(addr, "USER_CTRL", REG_USER_CTRL);
  printReg(addr, "CONFIG", REG_CONFIG);
  printReg(addr, "ACCEL_CONFIG", REG_ACCEL_CONFIG);
  printReg(addr, "GYRO_CONFIG", REG_GYRO_CONFIG);
  printRawSample(addr);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("MPU register debug");
  Serial.println("before setup");
  printStatus(MPU1_ADDR, "MPU1");
  printStatus(MPU2_ADDR, "MPU2");

  Serial.println("after original setup");
  setupMpuOriginal(MPU1_ADDR);
  setupMpuOriginal(MPU2_ADDR);
  printStatus(MPU1_ADDR, "MPU1");
  printStatus(MPU2_ADDR, "MPU2");

  Serial.println("after reset setup");
  setupMpuReset(MPU1_ADDR);
  setupMpuReset(MPU2_ADDR);
  printStatus(MPU1_ADDR, "MPU1");
  printStatus(MPU2_ADDR, "MPU2");
}

void loop() {
  static uint32_t lastPrint = 0;
  uint32_t now = millis();
  if (now - lastPrint < 1000) {
    return;
  }
  lastPrint = now;

  Serial.println("live raw");
  printRawSample(MPU1_ADDR);
  printRawSample(MPU2_ADDR);
}
