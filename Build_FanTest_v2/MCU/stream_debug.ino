#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define SDA_PIN 21
#define SCL_PIN 22

#define DS18B20_PIN 4
#define TEMP_CONVERSION_TIME 750
OneWire oneWire(DS18B20_PIN);
DallasTemperature sensors(&oneWire);

#define STREAM_BAUD 1000000
#define SAMPLE_PERIOD_US 1000

#define MPU1_ADDR 0x68
#define MPU2_ADDR 0x69

#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_INT_ENABLE   0x38
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1   0x6B

volatile float temperatureC = 0.0;
uint32_t nextTick = 0;
uint32_t lastPrint = 0;
uint32_t mpu1Ok = 0;
uint32_t mpu1Fail = 0;
uint32_t mpu2Ok = 0;
uint32_t mpu2Fail = 0;
uint32_t bothOk = 0;

void writeMPU(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
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
  Wire.setClock(100000);

  setupMPU(MPU1_ADDR);
  setupMPU(MPU2_ADDR);

  Serial.println("Stream debug. Waiting for START...");
  Serial.print("Sample struct size (bytes): ");
  Serial.println(32);
  waitForStart();
  Serial.println("START received");
  nextTick = micros();
  lastPrint = millis();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextTick) < 0) {
    return;
  }
  nextTick += SAMPLE_PERIOD_US;

  bool ok1 = readMPU14(MPU1_ADDR);
  bool ok2 = readMPU14(MPU2_ADDR);

  if (ok1) {
    mpu1Ok++;
  } else {
    mpu1Fail++;
  }

  if (ok2) {
    mpu2Ok++;
  } else {
    mpu2Fail++;
  }

  if (ok1 && ok2) {
    bothOk++;
  }

  if (millis() - lastPrint >= 1000) {
    lastPrint = millis();
    Serial.print("counts ");
    Serial.print("mpu1Ok=");
    Serial.print(mpu1Ok);
    Serial.print(" mpu1Fail=");
    Serial.print(mpu1Fail);
    Serial.print(" mpu2Ok=");
    Serial.print(mpu2Ok);
    Serial.print(" mpu2Fail=");
    Serial.print(mpu2Fail);
    Serial.print(" bothOk=");
    Serial.print(bothOk);
    Serial.print(" tempC=");
    Serial.println(temperatureC);
  }
}
