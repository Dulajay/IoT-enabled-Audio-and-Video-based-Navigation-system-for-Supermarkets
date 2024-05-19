// Include the SoftwareSerial library to communicate with the Bluetooth module
#include <SoftwareSerial.h>

// Define the pins for motor control
const int motor1Pin1 = 3; // Motor 1 input pin 1
const int motor1Pin2 = 5; // Motor 1 input pin 2
const int motor2Pin1 = 6; // Motor 2 input pin 1
const int motor2Pin2 = 9; // Motor 2 input pin 2

// Define the Bluetooth module's RX and TX pins
const int bluetoothRX = 7; // Bluetooth module RX pin connected to Arduino TX pin
const int bluetoothTX = 8; // Bluetooth module TX pin connected to Arduino RX pin

// Create a SoftwareSerial object to communicate with the Bluetooth module
SoftwareSerial bluetooth(bluetoothRX, bluetoothTX);

void setup() {
  // Set motor control pins as outputs
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);

  // Initialize serial communication for debugging
  Serial.begin(9600);

  // Initialize Bluetooth communication
  bluetooth.begin(38400);
}

void loop() {
  // Check if data is available from Bluetooth module
  if (bluetooth.available() > 0) {
    // Read the incoming byte
    char command = bluetooth.read();

    // Print the received command for debugging
    Serial.print("Received Command: ");
    Serial.println(command);

    // Check the received command and control motors accordingly
    switch (command) {
      case 'F': // Forward
        forward();
        break;
      case 'B': // Backward
        backward();
        break;
      case 'L': // Left (rotate left)
        left();
        break;
      case 'R': // Right (rotate right)
        right();
        break;
      case 'S': // Stop
        stop();
        break;
      default:
        // Invalid command, do nothing
        break;
    }
  }
}

// Function to move both motors forward
void forward() {
  Serial.print("forward");
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
}

// Function to move both motors backward
void backward() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
}

// Function to rotate left
void left() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
}

// Function to rotate right
void right() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
}

// Function to stop both motors
void stop() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
}
