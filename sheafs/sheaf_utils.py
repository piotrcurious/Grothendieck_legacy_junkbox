import random
import time

def lfsr_gen(seed, poly, bits=8):
    state = seed
    while True:
        yield state & 1
        feedback = 0
        for i in range(bits):
            if (poly >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << (bits - 1))

class MockSerial:
    def __init__(self, use_lfsr=True):
        self.in_waiting = 1
        self.time = 0
        self.use_lfsr = use_lfsr
        if self.use_lfsr:
            self.poly = 0b10110001
            self.seed = random.randint(1, 255)
            self.gen = lfsr_gen(self.seed, self.poly)
        print(f"MockSerial initialized (LFSR={use_lfsr})")

    def readline(self):
        time.sleep(0.05)
        self.time += random.randint(50, 500)
        if self.use_lfsr:
            value = next(self.gen)
        else:
            value = random.randint(0, 1)
        line = f"Collected Data Point: Time: {self.time} ms, Value: {value}\n"
        return line.encode('utf-8')

    def flushInput(self): pass
    def close(self): pass

def parse_data_line(line):
    """
    Parses a line of data from the Arduino.
    Format: "Collected Data Point: Time: <t> ms, Value: <v>"
    """
    if "Collected Data Point" in line:
        try:
            parts = line.split(',')
            time_part = parts[0].split('Time: ')[1].strip().replace(' ms', '')
            value_part = parts[1].split('Value: ')[1].strip()
            timestamp = int(time_part)
            value = int(value_part)
            return timestamp, value
        except (IndexError, ValueError):
            print(f"Failed to parse data line: {line}")
    return None, None
