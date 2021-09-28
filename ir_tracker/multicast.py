import time
import socket
import json

GROUP = "239.0.0.22"
PORT = 7071

HOPS = 2


class Broadcaster:
    def __init__(self, group=GROUP, port=PORT):
        self.group = group
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                    socket.IPPROTO_UDP)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL,
                               HOPS)

    def send(self, message):
        payload = bytes(message, 'utf-8')
        self.socket.sendto(payload, (self.group, self.port))

    def send_json(self, data):
        payload = json.dumps(data)
        self.send(payload)


if __name__ == "__main__":
    broadcaster = Broadcaster()
    while True:
        broadcaster.send_json({"test": "value"})
        time.sleep(0.5)
