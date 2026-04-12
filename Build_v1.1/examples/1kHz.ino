#include <Wire.h>

#define MPU_ADDR 0x68

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_INT_ENABLE   0x38
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1   0x6B

struct Sample {
  uint32_t t_us;
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};

const int BUFFER_SAMPLES = 32;
Sample buffer[BUFFER_SAMPLES];
volatile int bufIndex = 0;

void writeMPU(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

bool readMPU14(int16_t &ax, int16_t &ay, int16_t &az,
               int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;

  uint8_t n = Wire.requestFrom(MPU_ADDR, (uint8_t)14);
  if (n != 14) return false;

  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); // skip temperature
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
  return true;
}

void setupMPU() {
  writeMPU(REG_PWR_MGMT_1, 0x01);
  delay(100);
  writeMPU(REG_CONFIG, 0x03);      // DLPF enabled
  writeMPU(REG_SMPLRT_DIV, 0x00);  // 1 kHz
  writeMPU(REG_GYRO_CONFIG, 0x00); // +-250 dps
  writeMPU(REG_ACCEL_CONFIG, 0x00);// +-2g
  writeMPU(REG_INT_ENABLE, 0x01);  // data ready interrupt enable
}

void setup() {
  Serial.begin(921600);   // use a high baud rate
  Wire.begin();
  Wire.setClock(400000);  // fast I2C
  setupMPU();
}

void loop() {
  static uint32_t lastTick = 0;

  if ((int32_t)(micros() - lastTick) >= 1000) {
    lastTick += 1000;

    int16_t ax, ay, az, gx, gy, gz;
    if (readMPU14(ax, ay, az, gx, gy, gz)) {
      buffer[bufIndex].t_us = micros();
      buffer[bufIndex].ax = ax;
      buffer[bufIndex].ay = ay;
      buffer[bufIndex].az = az;
      buffer[bufIndex].gx = gx;
      buffer[bufIndex].gy = gy;
      buffer[bufIndex].gz = gz;
      bufIndex++;

      if (bufIndex >= BUFFER_SAMPLES) {
        Serial.write((uint8_t*)buffer, sizeof(buffer));
        bufIndex = 0;
      }
    }
  }
}
