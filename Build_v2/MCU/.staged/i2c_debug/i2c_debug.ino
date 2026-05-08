#include <Wire.h>

#define SDA_PIN 21
#define SCL_PIN 22

#define STREAM_BAUD 1000000
#define MPU1_ADDR 0x68
#define MPU2_ADDR 0x69

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_INT_ENABLE   0x38
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1   0x6B
#define REG_WHO_AM_I     0x75

uint32_t mpu1Ok = 0;
uint32_t mpu1Fail = 0;
uint32_t mpu2Ok = 0;
uint32_t mpu2Fail = 0;

bool pingAddr(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

void writeMPU(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool readReg(uint8_t addr, uint8_t reg, uint8_t &val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) {
    return false;
  }
  val = Wire.read();
  return true;
}

bool readMPU14(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  if (Wire.requestFrom(addr, (uint8_t)14) != 14) {
    return false;
  }
  while (Wire.available() > 0) {
    Wire.read();
  }
  return true;
}

void setupMPU(uint8_t addr) {
  writeMPU(addr, REG_PWR_MGMT_1, 0x01);
  delay(100);
  writeMPU(addr, REG_CONFIG, 0x03);
  writeMPU(addr, REG_SMPLRT_DIV, 0x00);
  writeMPU(addr, REG_GYRO_CONFIG, 0x00);
  writeMPU(addr, REG_ACCEL_CONFIG, 0x00);
  writeMPU(addr, REG_INT_ENABLE, 0x01);
}

void printScan() {
  Serial.println("I2C scan:");
  for (uint8_t addr = 1; addr < 127; addr++) {
    if (pingAddr(addr)) {
      Serial.print("  0x");
      if (addr < 16) {
        Serial.print("0");
      }
      Serial.println(addr, HEX);
    }
  }
}

void printMpuStatus(uint8_t addr, const char *name) {
  uint8_t who = 0;
  bool ping = pingAddr(addr);
  bool whoOk = readReg(addr, REG_WHO_AM_I, who);
  Serial.print(name);
  Serial.print(" addr=0x");
  Serial.print(addr, HEX);
  Serial.print(" ping=");
  Serial.print(ping ? "ok" : "fail");
  Serial.print(" who=");
  if (whoOk) {
    Serial.print("0x");
    Serial.print(who, HEX);
  } else {
    Serial.print("fail");
  }
  Serial.println();
}

void setup() {
  Serial.begin(STREAM_BAUD);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("I2C debug firmware");
  Serial.print("SDA=");
  Serial.print(SDA_PIN);
  Serial.print(" SCL=");
  Serial.print(SCL_PIN);
  Serial.print(" WireClock=");
  Serial.println(100000);

  printScan();
  printMpuStatus(MPU1_ADDR, "MPU1");
  printMpuStatus(MPU2_ADDR, "MPU2");

  if (pingAddr(MPU1_ADDR)) {
    setupMPU(MPU1_ADDR);
  }
  if (pingAddr(MPU2_ADDR)) {
    setupMPU(MPU2_ADDR);
  }
}

void loop() {
  if (readMPU14(MPU1_ADDR)) {
    mpu1Ok++;
  } else {
    mpu1Fail++;
  }

  if (readMPU14(MPU2_ADDR)) {
    mpu2Ok++;
  } else {
    mpu2Fail++;
  }

  Serial.print("readCounts ");
  Serial.print("mpu1Ok=");
  Serial.print(mpu1Ok);
  Serial.print(" mpu1Fail=");
  Serial.print(mpu1Fail);
  Serial.print(" mpu2Ok=");
  Serial.print(mpu2Ok);
  Serial.print(" mpu2Fail=");
  Serial.println(mpu2Fail);

  delay(1000);
}
