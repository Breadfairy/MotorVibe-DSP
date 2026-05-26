#include <Arduino.h>
#line 1 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
#include <Wire.h>
#include <MPU6050.h>

// ---------- Timing debug ----------
#define TIMING_DEBUG 0
#define TIMING_DEBUG_TEXT_ONLY 0
#define TIMING_DEBUG_BAUD 115200
#define STREAM_BAUD 2000000

// ---------- I2C ----------
#define SDA_PIN 4
#define SCL_PIN 5

MPU6050 imu1(0x68); // AD0 → GND
MPU6050 imu2(0x69); // AD0 → VCC

// ---------- Timing ----------
const unsigned long interval = 1250; // 800 Hz = 1250 µs
unsigned long lastMicros = 0;

uint32_t sample = 0;

// ---------- RAM buffer ----------
#define SAMPLE_SIZE 30
#define BATCH_SAMPLE_COUNT 800
#define BUFFER_SIZE (SAMPLE_SIZE * BATCH_SAMPLE_COUNT)
uint8_t buffer[BUFFER_SIZE];
uint32_t bufferIndex = 0;

// ---------- Header for each batch ----------
const uint8_t HEADER[4] = {0xAA, 0xBB, 0xCC, 0xDD};

// ---------- Runtime timing ----------
#if TIMING_DEBUG
unsigned long maxLoopMicros = 0;
unsigned long maxImuOneMicros = 0;
unsigned long maxImuTwoMicros = 0;
unsigned long maxFlushMicros = 0;
uint32_t lateSampleCount = 0;
#endif

// ---------- Setup ----------
#line 43 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
void setup();
#line 63 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
void sendInt16ToBuffer(int16_t val);
#line 68 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
void sendUint32ToBuffer(uint32_t val);
#line 85 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
void flushBuffer();
#line 104 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
void printTimingDebug();
#line 121 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
void loop();
#line 43 "/Users/broderickmadden-scott/programming/py/DSP-Motor/quickcapture/mcu/dataStreamBinary/dataStreamBinary.ino"
void setup() {
#if TIMING_DEBUG
    Serial.begin(TIMING_DEBUG_BAUD);
#else
    Serial.begin(STREAM_BAUD);
#endif
    delay(1000);

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(800000);

    imu1.initialize();
    imu2.initialize();
    lastMicros = micros();
#if TIMING_DEBUG
    Serial.println("timingDebugBooted");
#endif
}

// ---------- Helper functions ----------
inline void sendInt16ToBuffer(int16_t val) {
    buffer[bufferIndex++] = val & 0xFF;
    buffer[bufferIndex++] = (val >> 8) & 0xFF;
}

inline void sendUint32ToBuffer(uint32_t val) {
    buffer[bufferIndex++] = val & 0xFF;
    buffer[bufferIndex++] = (val >> 8) & 0xFF;
    buffer[bufferIndex++] = (val >> 16) & 0xFF;
    buffer[bufferIndex++] = (val >> 24) & 0xFF;
}

// Stores the current maximum timing value.
#if TIMING_DEBUG
void updateMax(unsigned long* maxValue, unsigned long currentValue) {
    if (currentValue > *maxValue) {
        *maxValue = currentValue;
    }
}
#endif

// Flush the buffer over Serial with header.
void flushBuffer() {
    if (bufferIndex > 0) {
#if TIMING_DEBUG_TEXT_ONLY
        bufferIndex = 0;
#else
#if TIMING_DEBUG
        unsigned long flushStart = micros();
#endif
        Serial.write(HEADER, sizeof(HEADER));
        Serial.write(buffer, bufferIndex);
        bufferIndex = 0;
#if TIMING_DEBUG
        updateMax(&maxFlushMicros, micros() - flushStart);
#endif
#endif
    }
}

// Prints the current worst-case timing values when enabled.
void printTimingDebug() {
#if TIMING_DEBUG
    Serial.println();
    Serial.print("maxLoopMicros:");
    Serial.println(maxLoopMicros);
    Serial.print("maxImuOneMicros:");
    Serial.println(maxImuOneMicros);
    Serial.print("maxImuTwoMicros:");
    Serial.println(maxImuTwoMicros);
    Serial.print("maxFlushMicros:");
    Serial.println(maxFlushMicros);
    Serial.print("lateSampleCount:");
    Serial.println(lateSampleCount);
#endif
}

// ---------- Main loop ----------
void loop() {
    unsigned long now = micros();

    while (now - lastMicros >= interval) {
#if TIMING_DEBUG
        unsigned long loopStart = micros();
        if (now - lastMicros >= (interval * 2)) {
            lateSampleCount++;
        }
#endif
        lastMicros += interval;

        // --- Read IMU1 ---
        int16_t ax1, ay1, az1, gx1, gy1, gz1;
#if TIMING_DEBUG
        unsigned long imuOneStart = micros();
#endif
        imu1.getMotion6(&ax1, &ay1, &az1, &gx1, &gy1, &gz1);
#if TIMING_DEBUG
        updateMax(&maxImuOneMicros, micros() - imuOneStart);
#endif

        // --- Read IMU2 ---
        int16_t ax2, ay2, az2, gx2, gy2, gz2;
#if TIMING_DEBUG
        unsigned long imuTwoStart = micros();
#endif
        imu2.getMotion6(&ax2, &ay2, &az2, &gx2, &gy2, &gz2);
#if TIMING_DEBUG
        updateMax(&maxImuTwoMicros, micros() - imuTwoStart);
#endif

        // --- Placeholder DS18B20 until the physical sensor is attached ---
        int16_t ds18b20 = 0;

        // --- Store sample in buffer ---
        sendUint32ToBuffer(sample++);
        sendInt16ToBuffer(ax1); sendInt16ToBuffer(ay1); sendInt16ToBuffer(az1);
        sendInt16ToBuffer(gx1); sendInt16ToBuffer(gy1); sendInt16ToBuffer(gz1);

        sendInt16ToBuffer(ax2); sendInt16ToBuffer(ay2); sendInt16ToBuffer(az2);
        sendInt16ToBuffer(gx2); sendInt16ToBuffer(gy2); sendInt16ToBuffer(gz2);
        sendInt16ToBuffer(ds18b20);

        // --- Flush one fixed 800-sample batch ---
        if (bufferIndex >= BUFFER_SIZE) {
            flushBuffer();
#if TIMING_DEBUG
            printTimingDebug();
#endif
        }

#if TIMING_DEBUG
        updateMax(&maxLoopMicros, micros() - loopStart);
#endif
        now = micros();
    }
}

