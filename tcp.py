import socket
from struct import unpack

# local port listener
HOST = "192.168.0.11"
PORT = 2001


def send_data(send_data, debug=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.connect((HOST, PORT))
        print('.', end='', flush=True) if not debug else None

        try:
            s.send(send_data)
            print('Sended', send_data) if debug else None
            print('S', end='', flush=True) if not debug else None
        except TimeoutError:
            print('.', end='', flush=True) if not debug else None
            print('Send Timeout') if debug else None
            return None


def receive_data(structure, debug=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.connect((HOST, PORT))
        print('.', end='', flush=True) if not debug else None

        try:
            rcv_data = s.recv(100)
            print('Received', rcv_data) if debug else None
            print('R', end='', flush=True) if not debug else None
            return unpack(structure, rcv_data)
        except TimeoutError:
            print('.', end='', flush=True) if not debug else None
            print('Receive Timeout') if debug else None
            return None
