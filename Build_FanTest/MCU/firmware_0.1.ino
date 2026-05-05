#include <Wire.h>

// ---------- I2C Pins ----------
#define SDA_PIN 4
#define SCL_PIN 5

// ---------- Transport ----------
#define STREAM_BAUD 1000000
#define FRAME_MAGIC 0xAABBCCDDUL
#define FRAME_VERSION 1

// ---------- Stream Modes ----------
#define MODE_ALL_1K 1
#define MODE_GYRO_8K 2

// ---------- MPU6050 ----------
#define MPU_ADDR 0x68

#define REG_SMPLRT_DIV      0x19
#define REG_CONFIG          0x1A
#define REG_GYRO_CONFIG     0x1B
#define REG_ACCEL_CONFIG    0x1C
#define REG_FIFO_EN         0x23
#define REG_INT_ENABLE      0x38
#define REG_ACCEL_XOUT_H    0x3B
#define REG_SIGNAL_PATH_RST 0x68
#define REG_USER_CTRL       0x6A
#define REG_PWR_MGMT_1      0x6B
#define REG_FIFO_COUNTH     0x72
#define REG_FIFO_R_W        0x74

// ---------- Framed Batch Header ----------
struct __attribute__((packed)) FrameHeader {
  uint32_t magic;
  uint8_t version;
  uint8_t mode;
  uint16_t sampleCount;
  uint32_t sequence;
  uint32_t startSample;
  uint32_t sampleRateHz;
  uint32_t payloadBytes;
};

const uint16_t ALL_1K_FRAME_SAMPLES = 128;
const uint16_t GYRO_FRAME_SAMPLES = 128;
const uint16_t ALL_1K_SAMPLE_BYTES = 12;
const uint16_t GYRO_SAMPLE_BYTES = 6;
const uint16_t FIFO_CAPACITY_BYTES = 1024;
const uint16_t FIFO_BURST_BYTES = 96;
const size_t MAX_PAYLOAD_BYTES =
  ALL_1K_FRAME_SAMPLES * ALL_1K_SAMPLE_BYTES;
const size_t MAX_FRAME_BYTES =
  sizeof(FrameHeader) + MAX_PAYLOAD_BYTES;

uint8_t frameA[MAX_FRAME_BYTES];
uint8_t frameB[MAX_FRAME_BYTES];
uint8_t *fillFrame = frameA;
uint8_t *sendFrame = frameB;
size_t fillOffset = sizeof(FrameHeader);
size_t sendOffset = 0;
size_t sendBytes = 0;
bool sendPending = false;

uint8_t streamMode = MODE_ALL_1K;
uint16_t frameSamples = ALL_1K_FRAME_SAMPLES;
uint16_t sampleBytes = ALL_1K_SAMPLE_BYTES;
uint32_t sampleRateHz = 1000;
uint32_t frameSequence = 0;
uint32_t sampleCounter = 0;
uint32_t frameStartSample = 0;
uint8_t pendingMode = MODE_ALL_1K;
bool streamActive = false;

void setStatusLed(bool sensing) {
#ifdef RGB_BUILTIN
  if (sensing) {
    rgbLedWrite(RGB_BUILTIN, 0, 32, 0);
    return;
  }
  rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
#endif
}

void writeMPU(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool readMPU(uint8_t reg, uint8_t *dst, size_t size) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(MPU_ADDR, (uint8_t)size) != size) {
    return false;
  }

  for (size_t i = 0; i < size; i++) {
    dst[i] = Wire.read();
  }
  return true;
}

void resetFIFO() {
  writeMPU(REG_FIFO_EN, 0x00);
  writeMPU(REG_USER_CTRL, 0x00);
  writeMPU(REG_USER_CTRL, 0x04);
  delay(1);
  writeMPU(REG_USER_CTRL, 0x40);
}

void configureFIFO(uint8_t fifoMask) {
  resetFIFO();
  writeMPU(REG_FIFO_EN, fifoMask);
}

uint16_t readFIFOCount() {
  uint8_t raw[2];
  if (!readMPU(REG_FIFO_COUNTH, raw, 2)) {
    return 0;
  }
  return ((uint16_t)raw[0] << 8) | raw[1];
}

bool readFIFOBurst(uint8_t *dst, size_t size) {
  size_t done = 0;
  while (done < size) {
    size_t count = size - done;
    if (count > FIFO_BURST_BYTES) {
      count = FIFO_BURST_BYTES;
    }
    if (!readMPU(REG_FIFO_R_W, dst + done, count)) {
      return false;
    }
    done += count;
  }
  return true;
}

void setupMPU() {
  writeMPU(REG_PWR_MGMT_1, 0x01);
  delay(100);
  writeMPU(REG_SIGNAL_PATH_RST, 0x07);
  delay(10);
  writeMPU(REG_GYRO_CONFIG, 0x00);
  writeMPU(REG_ACCEL_CONFIG, 0x00);
  writeMPU(REG_INT_ENABLE, 0x01);
  resetFIFO();
}

void resetFrames() {
  fillFrame = frameA;
  sendFrame = frameB;
  fillOffset = sizeof(FrameHeader);
  sendOffset = 0;
  sendBytes = 0;
  sendPending = false;
  frameSequence = 0;
  sampleCounter = 0;
  frameStartSample = 0;
}

void applyMode(uint8_t mode) {
  streamMode = mode;

  if (streamMode == MODE_ALL_1K) {
    frameSamples = ALL_1K_FRAME_SAMPLES;
    sampleBytes = ALL_1K_SAMPLE_BYTES;
    sampleRateHz = 1000;
    writeMPU(REG_CONFIG, 0x03);
    writeMPU(REG_SMPLRT_DIV, 0x00);
    configureFIFO(0x78);
  }

  if (streamMode == MODE_GYRO_8K) {
    frameSamples = GYRO_FRAME_SAMPLES;
    sampleBytes = GYRO_SAMPLE_BYTES;
    sampleRateHz = 8000;
    writeMPU(REG_CONFIG, 0x00);
    writeMPU(REG_SMPLRT_DIV, 0x00);
    configureFIFO(0x70);
  }

  resetFrames();
}

void startPendingStream() {
  applyMode(pendingMode);
  streamActive = true;
  setStatusLed(true);
}

void queueFrame() {
  FrameHeader header;
  header.magic = FRAME_MAGIC;
  header.version = FRAME_VERSION;
  header.mode = streamMode;
  header.sampleCount = frameSamples;
  header.sequence = frameSequence++;
  header.startSample = frameStartSample;
  header.sampleRateHz = sampleRateHz;
  header.payloadBytes = fillOffset - sizeof(FrameHeader);
  memcpy(fillFrame, &header, sizeof(FrameHeader));

  uint8_t *swapFrame = sendFrame;
  sendFrame = fillFrame;
  fillFrame = swapFrame;
  sendBytes = sizeof(FrameHeader) + header.payloadBytes;
  sendOffset = 0;
  sendPending = true;
  fillOffset = sizeof(FrameHeader);
}

void serviceSerial() {
  if (!sendPending) {
    return;
  }

  int writeSpace = Serial.availableForWrite();
  if (writeSpace <= 0) {
    return;
  }

  size_t remain = sendBytes - sendOffset;
  size_t writeBytes = writeSpace;
  if (writeBytes > remain) {
    writeBytes = remain;
  }

  sendOffset += Serial.write(sendFrame + sendOffset, writeBytes);

  if (sendOffset >= sendBytes) {
    sendOffset = 0;
    sendBytes = 0;
    sendPending = false;
  }
}

void handleCommand(String command) {
  if (command == "MODE ALL_1K") {
    pendingMode = MODE_ALL_1K;
  }

  if (command == "MODE GYRO_8K") {
    pendingMode = MODE_GYRO_8K;
  }

  if (command == "STOP") {
    streamActive = false;
    resetFrames();
    resetFIFO();
    setStatusLed(false);
  }

  if (command == "START") {
    startPendingStream();
  }
}

void serviceCommands() {
  while (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    handleCommand(command);
  }
}

void setup() {
  Serial.begin(STREAM_BAUD);
  Serial.setTimeout(10);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(1000000);

  setupMPU();
  applyMode(MODE_ALL_1K);
  setStatusLed(false);

  Serial.println("Ready. Waiting for MODE and START...");
  Serial.print("Frame header size (bytes): ");
  Serial.println(sizeof(FrameHeader));
  Serial.println("Modes: ALL_1K GYRO_8K");
}

void loop() {
  serviceCommands();

  if (!streamActive) {
    delay(1);
    return;
  }

  if (fillOffset >= sizeof(FrameHeader) + (frameSamples * sampleBytes)) {
    if (sendPending) {
      serviceSerial();
      return;
    }
    queueFrame();
  }

  serviceSerial();

  if (sendPending && fillOffset >= sizeof(FrameHeader) +
    (frameSamples * sampleBytes)) {
    return;
  }

  if (fillOffset == sizeof(FrameHeader)) {
    frameStartSample = sampleCounter;
  }

  uint16_t fifoCount = readFIFOCount();
  if (fifoCount >= (FIFO_CAPACITY_BYTES - sampleBytes)) {
    resetFIFO();
    return;
  }

  uint16_t availableSamples = fifoCount / sampleBytes;
  if (availableSamples == 0) {
    return;
  }

  uint16_t frameCount = (fillOffset - sizeof(FrameHeader)) / sampleBytes;
  uint16_t remainSamples = frameSamples - frameCount;
  uint16_t readSamples = availableSamples;
  if (readSamples > remainSamples) {
    readSamples = remainSamples;
  }

  size_t readBytes = readSamples * sampleBytes;
  if (!readFIFOBurst(fillFrame + fillOffset, readBytes)) {
    return;
  }

  fillOffset += readBytes;
  sampleCounter += readSamples;
}
