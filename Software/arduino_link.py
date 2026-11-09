import time
import serial

port = "/dev/ttyACM0"   # check with: ls /dev/ttyACM*
baud = 115200


def open_link():
    link = serial.Serial(port, baud)
    time.sleep(2)  # Arduino restarts when the port opens
    return link


def send(link, angles):
    message = ""
    for i in range(6):
        message = message + str(round(angles[i], 1))
        if i < 5:
            message = message + ","
    message = message + "\n"
    link.write(message.encode())
