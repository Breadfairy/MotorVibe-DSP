uint32_t sampleIndex = 0;


// Builds one signed dummy phase value that changes each sample.
int32_t buildPhase(uint32_t sampleIndex)
{
  return (int32_t)(sampleIndex % 50) - 25;
}


// Writes one full CSV row that matches the Python live sensor schema.
void sendDummyRow(uint32_t sampleIndex)
{
  int32_t phase = buildPhase(sampleIndex);
  int32_t heat = (int32_t)(sampleIndex % 20);

  Serial.print(sampleIndex);
  Serial.print(",");
  Serial.print(phase / 100.0, 2);
  Serial.print(",");
  Serial.print((-phase / 2) / 100.0, 2);
  Serial.print(",");
  Serial.print((981 + phase) / 100.0, 2);
  Serial.print(",");
  Serial.print((20 + (phase * 3)) / 100.0, 2);
  Serial.print(",");
  Serial.print((-10 + (phase * 2)) / 100.0, 2);
  Serial.print(",");
  Serial.print((5 - phase) / 100.0, 2);
  Serial.print(",");
  Serial.print((3150 + heat) / 100.0, 2);
  Serial.print(",");
  Serial.print((-phase) / 100.0, 2);
  Serial.print(",");
  Serial.print((phase / 3) / 100.0, 2);
  Serial.print(",");
  Serial.print((979 - phase) / 100.0, 2);
  Serial.print(",");
  Serial.print((18 - (phase * 2)) / 100.0, 2);
  Serial.print(",");
  Serial.print((8 + phase) / 100.0, 2);
  Serial.print(",");
  Serial.print((-6 - phase) / 100.0, 2);
  Serial.print(",");
  Serial.print((3140 + heat) / 100.0, 2);
  Serial.print(",");
  Serial.print((2810 + (heat / 2)) / 100.0, 2);
  Serial.print(",");
  Serial.println((2800 + (heat / 3)) / 100.0, 2);
}


// Starts the ESP32 serial link for host-side CSV reads.
void setup()
{
  Serial.begin(115200);
  delay(1000);
}


// Sends one dummy CSV row every 10 ms.
void loop()
{
  sendDummyRow(sampleIndex);
  sampleIndex++;
  delay(10);
}
