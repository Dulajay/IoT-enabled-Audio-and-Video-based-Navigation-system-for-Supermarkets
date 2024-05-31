#include <SoftwareSerial.h>
#include <string.h>

// serial communication packet specification
/* 
PACKET STRUCTURE

    <HEAD> <ANGLE> <SPEED_LSB> <SPEED_MSB> <16 BYTES FOR DATA> <CHECKSUM_LSB> <CHECKSUM_MSB>

    Of which inside the 16 bytes of data, there are four sets of the following bytes: 

    <DISTANCE_LSB><DISTANCE_MSB><SIGNAL_STR_LSB><SIGNAL_STR_MSB>

*/

// index of the bytes
#define ANGLE_IDX 1
#define SPEED_LSB 2
#define SPEED_MSB 3
#define DATA_1 4
#define DATA_2 8
#define DATA_3 12
#define DATA_4 16 
#define CHECKSUM_LSB 20
#define CHECKSUM_MSB 21

#define PACKET_SIZE 22  // size of packet
#define DATA_SIZE 7 // angle, speed, distance x 4, irradiance, validity

#define RX_PIN 10       // rx pin from the sensor
#define TX_PIN 11       // tx pin for software serial
#define BAUDRATE_SENSOR 115200  // baudrate of the sensor
#define BAUDRATE 115200
#define MIN_POWER 0     // minimum power of the motor
#define MAX_POWER 200   // maximum power of the motor
#define MOTOR_PIN 11     // motor pin number on Arduino Uno
#define MOTOR_SPEED 250 // motor speed in RPM

// Define the pins for motor control
const int motor1Pin1 = 6; // Motor 1 input pin 1
const int motor1Pin2 = 9; // Motor 1 input pin 2
const int motor2Pin1 = 3; // Motor 2 input pin 1
const int motor2Pin2 = 5; // Motor 2 input pin 2

// Define the Bluetooth module's RX and TX pins
const int bluetoothRX = 7; // Bluetooth module RX pin connected to Arduino TX pin
const int bluetoothTX = 8; // Bluetooth module TX pin connected to Arduino RX pin

int data[DATA_SIZE]; // [angle, speed, distance 1, distance 2, distance 3, distance 4, checksum]
uint8_t packet[PACKET_SIZE];    // packet buffer
const unsigned char HEAD_BYTE = 0xFA;   // start byte of the packet
unsigned int packetIndex      = 0;      // packet index
uint8_t receivedByte    = 0;    // received byte
uint8_t lowStrengthFlag = 0; 
bool PACKET_OK  = true;         // if the packet is valid
bool waitPacket = true;         // true if waiting for a packet

// PID control variables
double proportionalTerm = 0;
double derivativeTerm   = 0; 
double integralTerm     = 0;
double previousSpeed    = 0;

int currentSpeed    = 0; 
int baseSpeed       = 0; 
double t_sample     = 64*65536/16000000.0; // pre-scaler of 64
double kp = 2;
double ki = 0.3; 
double kd = 0.3;
int controlEffort = 0; 

// sensor ouput data
int angle = 0; 
int speed = 0; 

// uint16_t dist_1, dist_2, dist_3, dist_4 = 0, 0, 0, 0; 
SoftwareSerial lidarSensor(RX_PIN, TX_PIN); // RX and TX pin for SoftwareSerial
SoftwareSerial bluetooth(bluetoothRX, bluetoothTX);

void setup() {
    // Set motor control pins as outputs
    pinMode(motor1Pin1, OUTPUT);
    pinMode(motor1Pin2, OUTPUT);
    pinMode(motor2Pin1, OUTPUT);
    pinMode(motor2Pin2, OUTPUT);

    // setup TIMER0 -> trigger timer ISR every approx. 0.2s
    noInterrupts();     
    TCCR1A = 0;
    TCCR1B = 0;
    TCCR1B |= (1 << CS11)|(1 << CS10);   
    TIMSK1 |= (1 << TOIE1); 
    interrupts();
    
    // setup serial LiDAR -> Arduino
    lidarSensor.begin(BAUDRATE_SENSOR);

    // setup serial Arduino -> PC
    pinMode(RX_PIN, INPUT);
    Serial.begin(BAUDRATE);

    // Initialize Bluetooth communication
    bluetooth.begin(9600);

    // setup motor
    analogWrite(MOTOR_PIN, MAX_POWER);   // kick start motor with MAX power
    delay(2000);
    analogWrite(MOTOR_PIN, 125);

    // initialize packet buffer
    for (int idx = 0; idx < PACKET_SIZE; idx++) packet[idx] = 0;    // initialize packet buffer
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

    // check if any packet if arrived
    if (lidarSensor.available() > 0) {
        receivedByte = lidarSensor.read();

        if (waitPacket) { // wait for a new packet to arrive
            if (receivedByte == HEAD_BYTE) {
                packetIndex = 0;    // initialise packet index
                waitPacket = false;
                packet[packetIndex++] = receivedByte;
            }
        } else {  // if currently receiving packet
            if (packet[0] == HEAD_BYTE) { // ensure the head of the packet is valid
                packet[packetIndex++] = receivedByte; // store received byte
                if (packetIndex >= PACKET_SIZE) { // if packet buffer is full
                    waitPacket = true; // wait for a new packet
                    decodePacket(packet, PACKET_SIZE); // process the packet
                    sendData(data, DATA_SIZE);       
                }
            }
        }
    }
}

// timer interrupt handler for motor speed control
ISR(TIMER1_OVF_vect) {
    motorSpeedPID(MOTOR_SPEED, currentSpeed, 0.262, kp, ki, kd);
}

void decodePacket(uint8_t packet[], int packetSize) {
    int data_idx = 0; 

    for (int idx = 0; idx < DATA_SIZE; idx++) data[idx] = 0;  // initialise data array

    for (int i = 0; i < packetSize; i++){
        if (i == 0) {   // header byte
            continue;
        } else if (i == 1) {
            uint16_t angle = (packet[i] - 0xA0) * 4;  // convert to values between 0 ~ 360
            if (angle > 360) return; 
            data[data_idx++] = angle;
        } else if (i == 2) {
            int speed = 0x00; 
            speed |= ( (packet[3] << 8) | packet[2]);     
            currentSpeed = abs(speed / 64 - currentSpeed) > 100 ? currentSpeed * 0.95 + (speed / 64) * 0.05 : speed / 64; 
            data[data_idx++] = currentSpeed;
        } else if (i == 4 || i == 8 || i == 12 || i == 16) {
            uint16_t distance = 0x00;
            distance |= ((packet[i + 1] & 0x3F) << 8) | packet[i]; 
            data[data_idx++] = distance;
        }
    }

    uint16_t chksum = checksum(packet, (unsigned int)(packet[PACKET_SIZE - 2] + (packet[PACKET_SIZE - 1] << 8)), PACKET_SIZE - 2);
    data[data_idx++] = chksum; 
}

int sendData(int data[], int dataSize) {
    for (int i = 0; i < dataSize; i++) {
        Serial.print(data[i]); 
        Serial.print('\t'); 
    }
    Serial.println();  
}

uint16_t checksum(uint8_t packet[], uint16_t sum, uint8_t size) {
    uint16_t sensorData[2000]; 
    unsigned long chk32 = 0;

    for (int i = 0; i < size; i++)
        sensorData[i] = packet[i];

    for (int i = 0; i < size / 2; i++) {
        data[i] = ((sensorData[i * 2 + 1] << 8) | sensorData[i * 2]);
        chk32 = (chk32 << 1) + data[i];
    }

    uint16_t checksum = (chk32 & 0x7FFF) + (chk32 >> 15);
    checksum = checksum & 0x7FFF; 
    return sum == checksum; 
}

void motorSpeedPID(double targetSpeed, double currentSpeed, double period, double kp, double ki, double kd) {
    proportionalTerm = targetSpeed - currentSpeed;
    integralTerm += proportionalTerm * period;
    derivativeTerm = (currentSpeed - previousSpeed) / period;

    controlEffort = kp * proportionalTerm + ki * integralTerm - kd * derivativeTerm; 
    controlEffort = controlEffort > 255 ? 255 : controlEffort;
    controlEffort = controlEffort < 0 ? 0 : controlEffort;

    previousSpeed = currentSpeed;

    analogWrite(MOTOR_PIN, controlEffort); 
}

// Function to move the robot forward
void forward() {
    digitalWrite(motor1Pin1, HIGH);
    digitalWrite(motor1Pin2, LOW);
    digitalWrite(motor2Pin1, HIGH);
    digitalWrite(motor2Pin2, LOW);
}

// Function to move the robot backward
void backward() {
    digitalWrite(motor1Pin1, LOW);
    digitalWrite(motor1Pin2, HIGH);
    digitalWrite(motor2Pin1, LOW);
    digitalWrite(motor2Pin2, HIGH);
}

// Function to turn the robot left
void left() {
    digitalWrite(motor1Pin1, LOW);
    digitalWrite(motor1Pin2, HIGH);
    digitalWrite(motor2Pin1, HIGH);
    digitalWrite(motor2Pin2, LOW);
}

// Function to turn the robot right
void right() {
    digitalWrite(motor1Pin1, HIGH);
    digitalWrite(motor1Pin2, LOW);
    digitalWrite(motor2Pin1, LOW);
    digitalWrite(motor2Pin2, HIGH);
}

// Function to stop the robot
void stop() {
    digitalWrite(motor1Pin1, LOW);
    digitalWrite(motor1Pin2, LOW);
    digitalWrite(motor2Pin1, LOW);
    digitalWrite(motor2Pin2, LOW);
}
