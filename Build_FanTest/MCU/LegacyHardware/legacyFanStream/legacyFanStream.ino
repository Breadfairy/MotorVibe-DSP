#include <Wire.h>

// ---------- I2C Pins ----------
#define SDA_PIN 4
#define SCL_PIN 5

// ---------- Stream Timing ----------
#define STREAM_BAUD 1000000
#define SAMPLE_PERIOD_US 1000

// ---------- MPU6050 ----------
#define MPU_ADDR 0x68

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_INT_ENABLE   0x38
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1   0x6B

// ---------- Packed Struct ----------
struct __attribute__((packed)) Sample {
  uint32_t t_us;
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
};

const int BUFFER_SAMPLES = 32;
Sample bufferA[BUFFER_SAMPLES];
Sample bufferB[BUFFER_SAMPLES];
Sample *fillBuffer = bufferA;
Sample *sendBuffer = bufferB;
int fillIndex = 0;
size_t sendOffset = 0;
bool sendPending = false;
uint32_t nextTick = 0;

void writeMPU(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool readMPU14(
  int16_t &ax,
  int16_t &ay,
  int16_t &az,
  int16_t &gx,
  int16_t &gy,
  int16_t &gz
) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) {
    return false;
  }

  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read();
  Wire.read();
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
  return true;
}

void setupMPU() {
  writeMPU(REG_PWR_MGMT_1, 0x01);
  delay(100);
  writeMPU(REG_CONFIG, 0x03);
  writeMPU(REG_SMPLRT_DIV, 0x00);
  writeMPU(REG_GYRO_CONFIG, 0x00);
  writeMPU(REG_ACCEL_CONFIG, 0x00);
  writeMPU(REG_INT_ENABLE, 0x01);
}

void waitForStart() {
  while (true) {
    if (Serial.available() <= 0) {
      delay(1);
      continue;
    }

    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "START") {
      return;
    }
  }
}

void serviceSerial() {
  if (!sendPending) {
    return;
  }

  int writeSpace = Serial.availableForWrite();
  if (writeSpace <= 0) {
    return;
  }

  size_t sendBytes = sizeof(Sample) * BUFFER_SAMPLES;
  size_t remainingBytes = sendBytes - sendOffset;
  size_t writeBytes = writeSpace;
  if (writeBytes > remainingBytes) {
    writeBytes = remainingBytes;
  }

  sendOffset += Serial.write(
    ((uint8_t *)sendBuffer) + sendOffset,
    writeBytes
  );

  if (sendOffset >= sendBytes) {
    sendOffset = 0;
    sendPending = false;
  }
}

void setup() {
  Serial.begin(STREAM_BAUD);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  setupMPU();

  Serial.println("Ready. Waiting for START...");
  Serial.print("Sample struct size (bytes): ");
  Serial.println(sizeof(Sample));
  waitForStart();
  nextTick = micros();
}

void loop() {
  if (fillIndex >= BUFFER_SAMPLES) {
    if (sendPending) {
      serviceSerial();
      return;
    }

    Sample *swapBuffer = sendBuffer;
    sendBuffer = fillBuffer;
    fillBuffer = swapBuffer;
    fillIndex = 0;
    sendOffset = 0;
    sendPending = true;
  }

  serviceSerial();

  uint32_t now = micros();
  if ((int32_t)(now - nextTick) < 0) {
    return;
  }

  nextTick += SAMPLE_PERIOD_US;

  int16_t ax, ay, az, gx, gy, gz;
  if (readMPU14(ax, ay, az, gx, gy, gz)) {
    Sample &sample = fillBuffer[fillIndex];
    sample.t_us = micros();
    sample.ax = ax;
    sample.ay = ay;
    sample.az = az;
    sample.gx = gx;
    sample.gy = gy;
    sample.gz = gz;
    fillIndex++;
  }
}
