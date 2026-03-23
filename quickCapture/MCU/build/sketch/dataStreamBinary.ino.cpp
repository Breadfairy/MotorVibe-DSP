#include <Arduino.h>
#line 1 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/dataStreamBinary/dataStreamBinary.ino"
#include <Wire.h>
#include <MPU6050.h>

// ---------- I2C ----------
#define SDA_PIN 4
#define SCL_PIN 5

MPU6050 imu1(0x68); // AD0 → GND
MPU6050 imu2(0x69); // AD0 → VCC

// ---------- Timing ----------
const unsigned long interval = 1000; // 1 kHz = 1000 µs
unsigned long lastMicros = 0;

uint32_t sample = 0;

// ---------- RAM buffer ----------
#define SAMPLE_SIZE 32
#define BATCH_SAMPLE_COUNT 1000
#define BUFFER_SIZE (SAMPLE_SIZE * BATCH_SAMPLE_COUNT)
uint8_t buffer[BUFFER_SIZE];
uint32_t bufferIndex = 0;

// ---------- Header for each batch ----------
const uint8_t HEADER[4] = {0xAA, 0xBB, 0xCC, 0xDD};

// ---------- Setup ----------
#line 28 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/dataStreamBinary/dataStreamBinary.ino"
void setup();
#line 41 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/dataStreamBinary/dataStreamBinary.ino"
void sendInt16ToBuffer(int16_t val);
#line 46 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/dataStreamBinary/dataStreamBinary.ino"
void sendUint32ToBuffer(uint32_t val);
#line 54 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/dataStreamBinary/dataStreamBinary.ino"
void flushBuffer();
#line 63 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/dataStreamBinary/dataStreamBinary.ino"
void loop();
#line 28 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/dataStreamBinary/dataStreamBinary.ino"
void setup() {
    Serial.begin(2000000); // 2 Mbps
    delay(1000);

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    imu1.initialize();
    imu2.initialize();
    lastMicros = micros();
}

// ---------- Helper functions ----------
void sendInt16ToBuffer(int16_t val) {
    buffer[bufferIndex++] = val & 0xFF;
    buffer[bufferIndex++] = (val >> 8) & 0xFF;
}

void sendUint32ToBuffer(uint32_t val) {
    buffer[bufferIndex++] = val & 0xFF;
    buffer[bufferIndex++] = (val >> 8) & 0xFF;
    buffer[bufferIndex++] = (val >> 16) & 0xFF;
    buffer[bufferIndex++] = (val >> 24) & 0xFF;
}

// Flush the buffer over Serial with header.
void flushBuffer() {
    if (bufferIndex > 0) {
        Serial.write(HEADER, sizeof(HEADER));
        Serial.write(buffer, bufferIndex);
        bufferIndex = 0;
    }
}

// ---------- Main loop ----------
void loop() {
    unsigned long now = micros();

    if (now - lastMicros >= interval) {
        lastMicros += interval;

        // --- Read IMU1 ---
        int16_t ax1, ay1, az1, gx1, gy1, gz1;
        imu1.getMotion6(&ax1, &ay1, &az1, &gx1, &gy1, &gz1);
        int16_t temp1 = 0;

        // --- Read IMU2 ---
        int16_t ax2, ay2, az2, gx2, gy2, gz2;
        imu2.getMotion6(&ax2, &ay2, &az2, &gx2, &gy2, &gz2);
        int16_t temp2 = 0;

        // --- Store sample in buffer ---
        sendUint32ToBuffer(sample++);
        sendInt16ToBuffer(ax1); sendInt16ToBuffer(ay1); sendInt16ToBuffer(az1);
        sendInt16ToBuffer(gx1); sendInt16ToBuffer(gy1); sendInt16ToBuffer(gz1);
        sendInt16ToBuffer(temp1);

        sendInt16ToBuffer(ax2); sendInt16ToBuffer(ay2); sendInt16ToBuffer(az2);
        sendInt16ToBuffer(gx2); sendInt16ToBuffer(gy2); sendInt16ToBuffer(gz2);
        sendInt16ToBuffer(temp2);

        // --- Flush one fixed 1000-sample batch ---
        if (bufferIndex >= BUFFER_SIZE) {
            flushBuffer();
        }
    }
}

