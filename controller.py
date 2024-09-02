from machine import UART, Pin
import struct

class IBus:
    def __init__(self, uart):
        self.uart = uart
        self.channels = [0] * 14

    def update(self):
        if self.uart.any() >= 32:
            data = self.uart.read(32)
            if data[0] == 0x20 and data[1] == 0x40:
                for i in range(14):
                    self.channels[i] = struct.unpack('<H', data[2 + i*2:4 + i*2])[0]

    def get_channel(self, channel):
        return self.channels[channel] if 0 <= channel < len(self.channels) else None
    
uart = UART(1, baudrate=115200, tx=Pin(9), rx=Pin(10))

ibus = IBus(uart=uart)

while True:
    ibus.update()
    ch1 = ibus.get_channel(0)
    ch2 = ibus.get_channel(1)
    print('Channel 1: {}'.format(ch1))
    print('Channel 2: {}'.format(ch2))