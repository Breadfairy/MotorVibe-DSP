#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ---------- I2C Pins ----------
#define SDA_PIN 21
#define SCL_PIN 22

// ---------- DS18B20 ----------
#define DS18B20_PIN 4
OneWire oneWire(DS18B20_PIN);
DallasTemperature sensors(&oneWire);

// ---------- MPU6050 Addresses ----------
#define MPU1_ADDR 0x68
#define MPU2_ADDR 0x69

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_INT_ENABLE   0x38
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1   0x6B

// ---------- Packed Sample Struct ----------
struct __attribute__((packed)) Sample {
  uint32_t t_us;
  int16_t ax1, ay1, az1;
  int16_t gx1, gy1, gz1;
  int16_t ax2, ay2, az2;
  int16_t gx2, gy2, gz2;
  float tempC;
};

const int BUFFER_SAMPLES = 32;
Sample buffer[BUFFER_SAMPLES];
volatile int bufIndex = 0;

// ---------- Helper Functions ----------
void writeMPU(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool readMPU14(uint8_t addr, int16_t &ax, int16_t &ay, int16_t &az,
               int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;

  if (Wire.requestFrom(addr, (uint8_t)14) != 14) return false;

  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); // skip temperature
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
  return true;
}

void setupMPU(uint8_t addr) {
  writeMPU(addr, REG_PWR_MGMT_1, 0x01);
  delay(100);
  writeMPU(addr, REG_CONFIG, 0x03);      // DLPF enabled
  writeMPU(addr, REG_SMPLRT_DIV, 0x00);  // 1 kHz
  writeMPU(addr, REG_GYRO_CONFIG, 0x00); // ±250 dps
  writeMPU(addr, REG_ACCEL_CONFIG, 0x00);// ±2g
  writeMPU(addr, REG_INT_ENABLE, 0x01);  // data ready interrupt
}

// ---------- Global Variables ----------
float temperatureC = 0;
unsigned long lastTempTime = 0;

void setup() {
  Serial.begin(921600);
  delay(1000);

  // DS18B20 setup
  pinMode(DS18B20_PIN, INPUT_PULLUP);
  sensors.begin();
  sensors.setWaitForConversion(false);
  sensors.requestTemperatures();
  lastTempTime = millis();

  // MPU6050 setup
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(1000000); // 1 MHz I2C

  setupMPU(MPU1_ADDR);
  setupMPU(MPU2_ADDR);

  Serial.println("Ready. Sending binary samples...");
  Serial.print("Sample struct size (bytes): ");
  Serial.println(sizeof(Sample)); // should print 32
}

void loop() {
  static uint32_t lastTick = 0;
  unsigned long now = micros();

  // ---- Temperature every 1 sec ----
  if (millis() - lastTempTime >= 1000) {
    temperatureC = sensors.getTempCByIndex(0);
    sensors.requestTemperatures(); // next conversion
    lastTempTime = millis();
  }

  // ---- Read IMUs at 1 kHz ----
  if ((int32_t)(now - lastTick) >= 1000) {
    lastTick += 1000;

    int16_t ax1, ay1, az1, gx1, gy1, gz1;
    int16_t ax2, ay2, az2, gx2, gy2, gz2;

    if (readMPU14(MPU1_ADDR, ax1, ay1, az1, gx1, gy1, gz1) &&
        readMPU14(MPU2_ADDR, ax2, ay2, az2, gx2, gy2, gz2)) {

      buffer[bufIndex].t_us = micros();
      buffer[bufIndex].ax1 = ax1; buffer[bufIndex].ay1 = ay1; buffer[bufIndex].az1 = az1;
      buffer[bufIndex].gx1 = gx1; buffer[bufIndex].gy1 = gy1; buffer[bufIndex].gz1 = gz1;
      buffer[bufIndex].ax2 = ax2; buffer[bufIndex].ay2 = ay2; buffer[bufIndex].az2 = az2;
      buffer[bufIndex].gx2 = gx2; buffer[bufIndex].gy2 = gy2; buffer[bufIndex].gz2 = gz2;
      buffer[bufIndex].tempC = temperatureC;

      bufIndex++;

      if (bufIndex >= BUFFER_SAMPLES) {
        Serial.write((uint8_t*)buffer, sizeof(buffer));
        bufIndex = 0;
      }
    }
  }
}