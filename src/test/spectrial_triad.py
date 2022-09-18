
# import serial
# import keyboard
# ser = serial.Serial("/dev/ttyUSB0", 115200)
# while True:
#      #try:
#      line=str(ser.readline())
#      print(line)
     # if keyboard.is_pressed('q'):
     #      print("Exiting")
     #      break
     # elif keyboard.is_pressed('s'):
     #      print("Saving to .csv")
     #      continue
     # except:
     #      print("Exception occured")
     #      break

import serial
from pynput import keyboard
import time
import csv

ser = serial.Serial("/dev/ttyUSB0", 115200)
RECORD = False
delay = 1
time_len  = 120
Data =[]
path = '/home/namal/Documents/'

def record( time_len ):
     global RECORD, delay
     t = time_len
     file = str(input('Enter file name: '))
     with open(path+file+'.csv', 'w+') as new_file:
         csv_writer = csv.writer(new_file)
         while t>0 and RECORD:
              line = str(ser.readline())
              csv_writer.writerow(line[2:-5].split(','))
              print(line[2:-5])
              time.sleep(delay)
              t-=1
         print('Recoreded until end')

def on_press(key):
     global RECORD, time_len, record
     try:
         if key.char == 'q':
              RECORD = False
              print(RECORD)
         elif key.char == 'r':
              RECORD = True
              record(time_len)
         else:
              print('wrong Key')
     except AttributeError:
        print('special key {0} pressed'.format(
            key))


# Collect events until released
with keyboard.Listener(on_press=on_press,) as listener:
    listener.join()


