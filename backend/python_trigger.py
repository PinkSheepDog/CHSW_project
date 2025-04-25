import serial
import time

# Replace 'COM3' with your actual port
PORT = "COM3"
BAUD = 9600

def send_severe():
    # open serial
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)             # give ESP‑32 time to reset and start Serial
    ser.write(b"severe\n")    # send the command
    print("➡️  Sent 'severe' to", PORT)
    ser.close()

if __name__ == "__main__":
    send_severe()
