import time

import paramiko
from scp import SCPClient
from os import listdir
from getpass import getpass

SERVER = 'wolfe.cdms.westernsydney.edu.au'
PORT = '22'
USER = 'njayasuriya'
PASSWORD = getpass("User Account : ", stream=None)

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

ssh = createSSHClient(SERVER, PORT, USER, PASSWORD)
scp = SCPClient(ssh.get_transport())


TYPE = 'bag'
DATE = '05_10_2022'
FROM_DIR = '../../data/'+TYPE+'_files/'+DATE+'/'
TO_DIR = 'Crop_Imaging/data/'+TYPE+'_files/'+DATE+'/'
UPLOAD = True
SELECTION = 'R6'

if UPLOAD:
    files = listdir(FROM_DIR)
    for f in files:
        if f.__contains__(SELECTION):
            source = FROM_DIR+f
            print(source)
            scp.put(source, TO_DIR, recursive=True)
            time.sleep(5)
else:
    scp.get(TO_DIR, FROM_DIR, recursive=True)