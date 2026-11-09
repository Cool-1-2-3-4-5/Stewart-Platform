#include <Servo.h>

int pins[6] = {3, 5, 6, 9, 10, 11};  // signal pin for servo 1 to 6
Servo servos[6];

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 6; i++) {
    servos[i].attach(pins[i]);
    servos[i].write(90);  // start level
  }
}

void loop() {
  if (Serial.available() > 0) {
    for (int i = 0; i < 6; i++) {
      float angle = Serial.parseFloat();  // reads the next number, skips the commas
      angle = constrain(angle, 0, 180);
      servos[i].write(angle);
    }
    Serial.readStringUntil('\n');  // throw away the end of the line
  }
}
