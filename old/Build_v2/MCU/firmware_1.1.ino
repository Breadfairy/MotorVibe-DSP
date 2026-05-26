#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ---------- I2C Pins ----------
#define SDA_PIN 21
#define SCL_PIN 22

// ---------- DS18B20 ----------
#define DS18B20_PIN 4
#define TEMP_CONVERSION_TIME 750
OneWire oneWire(DS18B20_PIN);
DallasTemperature sensors(&oneWire);

// ---------- Stream Timing ----------
#define STREAM_BAUD 1000000
#define SAMPLE_PERIOD_US 1000

// ---------- MPU6050 ----------
#define MPU1_ADDR 0x68
#define MPU2_ADDR 0x69

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_INT_ENABLE   0x38
#define REG_ACCEL_XOUT_H 0x3B
#define REG_SIGNAL_RESET 0x68
#define REG_USER_CTRL    0x6A
#define REG_PWR_MGMT_1   0x6B
#define REG_PWR_MGMT_2   0x6C

// ---------- Packed Struct ----------
struct __attribute__((packed)) Sample {
  uint32_t t_us;
  int16_t ax1, ay1, az1;
  int16_t gx1, gy1, gz1;
  int16_t ax2, ay2, az2;
  int16_t gx2, gy2, gz2;
  float tempC;
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

// ---------- Temperature State ----------
volatile float temperatureC = 0.0;

// ---------- MPU Helpers ----------
void writeMPU(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool readMPU14(
  uint8_t addr,
  int16_t &ax,
  int16_t &ay,
  int16_t &az,
  int16_t &gx,
  int16_t &gy,
  int16_t &gz
) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(addr, (uint8_t)14) != 14) {
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

void setupMPU(uint8_t addr) {
  writeMPU(addr, REG_PWR_MGMT_1, 0x80);
  delay(100);
  writeMPU(addr, REG_SIGNAL_RESET, 0x07);
  delay(20);
  writeMPU(addr, REG_PWR_MGMT_1, 0x01);
  writeMPU(addr, REG_PWR_MGMT_2, 0x00);
  writeMPU(addr, REG_USER_CTRL, 0x00);
  writeMPU(addr, REG_CONFIG, 0x03);
  writeMPU(addr, REG_SMPLRT_DIV, 0x00);
  writeMPU(addr, REG_GYRO_CONFIG, 0x00);
  writeMPU(addr, REG_ACCEL_CONFIG, 0x00);
  writeMPU(addr, REG_INT_ENABLE, 0x01);
  delay(100);
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

void tempTask(void *taskData) {
  sensors.requestTemperatures();

  while (true) {
    vTaskDelay(pdMS_TO_TICKS(TEMP_CONVERSION_TIME));
    float t = sensors.getTempCByIndex(0);
    if (t > -40.0 && t < 125.0) {
      temperatureC = t;
    }
    sensors.requestTemperatures();
  }
}

void setup() {
  Serial.begin(STREAM_BAUD);
  delay(1000);

  sensors.begin();
  sensors.setWaitForConversion(false);
  xTaskCreatePinnedToCore(
    tempTask,
    "tempTask",
    4096,
    NULL,
    1,
    NULL,
    0
  );

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(1000000);

  setupMPU(MPU1_ADDR);
  setupMPU(MPU2_ADDR);

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

  int16_t ax1, ay1, az1, gx1, gy1, gz1;
  int16_t ax2, ay2, az2, gx2, gy2, gz2;

  if (
    readMPU14(MPU1_ADDR, ax1, ay1, az1, gx1, gy1, gz1) &&
    readMPU14(MPU2_ADDR, ax2, ay2, az2, gx2, gy2, gz2)
  ) {
    Sample &s = fillBuffer[fillIndex];
    s.t_us = micros();
    s.ax1 = ax1;
    s.ay1 = ay1;
    s.az1 = az1;
    s.gx1 = gx1;
    s.gy1 = gy1;
    s.gz1 = gz1;
    s.ax2 = ax2;
    s.ay2 = ay2;
    s.az2 = az2;
    s.gx2 = gx2;
    s.gy2 = gy2;
    s.gz2 = gz2;
    s.tempC = temperatureC;
    fillIndex++;
  }
}
