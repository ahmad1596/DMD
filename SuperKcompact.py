# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import serial
import time

class SuperKCompactController:
    def __init__(self, port, baudrate=115200):
        self.serial_port = serial.Serial(port, baudrate, timeout=1)
        self.addr_laser = '01'
        self.msg_write = '01'

    def send_telegram(self, addr, msg_type, data):
        telegram = f'{addr}{msg_type}{data}\r'
        print("Sending telegram:", telegram)
        self.serial_port.write(telegram.encode())

    def get_response(self):
        response = self.serial_port.read_until(b'\n').decode().strip()
        print("Received response:", response)
        return response

    def turn_on(self):
        data = '30' + '01' 
        self.send_telegram(self.addr_laser, self.msg_write, data)
        print("Laser turned ON")
        self.get_response()  

    def turn_off(self):
        data = '30' + '00' 
        self.send_telegram(self.addr_laser, self.msg_write, data)
        print("Laser turned OFF")
        self.get_response()  

    def close(self):
        self.serial_port.close()

def main():
    laser_controller = SuperKCompactController(port='COM3')  
    try:
        laser_controller.turn_on() 
        time.sleep(2)  
        laser_controller.turn_off() 
    finally:
        laser_controller.close() 
if __name__ == "__main__":
    main()
