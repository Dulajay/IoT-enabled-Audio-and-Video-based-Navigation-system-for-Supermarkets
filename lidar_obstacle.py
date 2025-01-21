import serial
import matplotlib.pyplot as plt
import statistics
import math
import time

class LidarData():
    
    def __init__(self):        
        self.DATA_LENGTH = 7        # data length: angle, speed, distance 1-4, checksum
        self.MAX_DISTANCE = 3000    # in mm
        self.MIN_DISTANCE = 100     # in mm
        self.port_lidar = 'COM14'    # Lidar serial port (Arduino Nano)
        self.port_motors = 'COM10'  # Motor control serial port (Arduino Mega)
        self.MAX_DATA_SIZE = 360    # resolution: 1 degree
        self.ser_lidar = None
        self.ser_motors = None
        self.BAUDRATE = 115200

        self.data = {   # sensor data 
            'angles'    : [],
            'distances' : [],
            'speed'     : [],
            'signal_strength' : [],  # TODO:
            'checksum'  : [] 
        }

        # Setup plot for Lidar visualization
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.ax.set_rmax(300)

        # Connect to the serial ports
        self.connectSerial(self.port_lidar, self.BAUDRATE, is_lidar=True)  # Lidar serial
        self.connectSerial(self.port_motors, 9600, is_lidar=False)  # Motor control serial

    def connectSerial(self, port: str, baudrate: int, is_lidar: bool) -> bool:
        try: 
            if is_lidar:
                self.ser_lidar = serial.Serial(port, baudrate, timeout=1)
                self.ser_lidar.reset_input_buffer()
                print(f'Lidar serial connection established @ {port}')
            else:
                self.ser_motors = serial.Serial(port, baudrate, timeout=1)
                self.ser_motors.reset_input_buffer()
                print(f'Motor control serial connection established @ {port}')
            return True  # return True to confirm connection
        except Exception as e: 
            print(e)
            return False
        
    def plotData(self) -> None:  # Plot data on a polar plot
        angles, distances = [], []
        
        for p in range(3, len(self.data['angles']) - 3):
            # Implement outlier filtering
            if (p > len(self.data['angles']) - 3): break
            sample = self.data['distances'][p-3:p+3]
            std = statistics.stdev(sample)
            if abs(self.data['distances'][p] - statistics.mean(sample)) < std:
                angles.append(self.data['angles'][p])
                distances.append(self.data['distances'][p])
            # ------------- Filtering END ---------------------

        self.ax.clear()  # Clear current plot
        plt.plot(angles, distances, ".")  # Plot points
        self.ax.set_rmax(self.MAX_DISTANCE)
        self.data['angles'].clear()
        self.data['distances'].clear()
        plt.draw()
        plt.pause(0.001)

    def updateData(self) -> None: 
        while True:
            try: 
                if self.ser_lidar and self.ser_lidar.in_waiting > 0:  # Check lidar serial connection
                    
                    try:  # try to read serial input; if unsuccessful, ignore the data
                        line = self.ser_lidar.readline().decode().rstrip()
                        sensorData = line.split('\t')  
                    except: 
                        continue
                    
                    # Process sensor data
                    if len(sensorData) == self.DATA_LENGTH:
                        valid_distances = []  # List for storing distances for averaging
                        
                        for i in range(2, 6):  # Split into four data points
                            try:
                                angle = (int(sensorData[0]) + i - 1) * math.pi / 180  # angle in radians
                                angle_degrees = round(angle * 180 / math.pi)  # Convert to degrees
                                dist = float(sensorData[i])  # Distance in mm
                                
                                # Only process angles between 0 and 10 degrees
                                if 0 <= angle_degrees <= 10:
                                    print(f'speed: {int(sensorData[1])} RPM, angle: {angle_degrees}, dist: {round(dist)}')
                                    valid_distances.append(dist)  # Store valid distance for averaging

                            except: 
                                continue
                            
                            # Update sensor data if valid
                            if self.MIN_DISTANCE <= dist <= self.MAX_DISTANCE:
                                self.data['angles'].append(angle)
                                self.data['distances'].append(dist)
                                self.data['checksum'].append(sensorData[-1])
                                self.data['speed'].append(sensorData[1])
                                
                            # If enough data is available, plot
                            if len(self.data['angles']) == self.MAX_DATA_SIZE:  
                                self.plotData()
                    
                        # Calculate average distance for angles between 0 and 20 degrees
                        if valid_distances:  
                            average_distance = sum(valid_distances) / len(valid_distances)
                            print(f'Average distance for angles 0 to 10 degrees: {average_distance:.2f} mm')

                            # Motor control based on average distance
                            if average_distance > 500:
                                self.sendMotorCommand('F')  # Move forward if distance > 350 mm
                            #elif 150 < average_distance <= 350:
                                #self.sendMotorCommand('L')  # Turn left for moderate distance
                            elif average_distance <= 500:
                                self.sendMotorCommand('S')  # Stop if distance <= 150 mm
                            
            except KeyboardInterrupt:
                exit()
    
    def sendMotorCommand(self, command: str) -> None:
        """Send a command to the motor controller via serial."""
        if self.ser_motors:
            try:
                self.ser_motors.write(command.encode())  # Send command to motors
                print(f"Sent command to motors: {command}")
            except Exception as e:
                print(f"Failed to send command: {e}")

    def getDistances(self) -> list: 
        return self.data['distances']
    
    def getAngles(self) -> list: 
        return self.data['angles']

if __name__ == '__main__':
    sensor = LidarData()
    sensor.updateData()
