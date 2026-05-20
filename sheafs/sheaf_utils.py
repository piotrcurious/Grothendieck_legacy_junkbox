import random
import time

class MockSerial:
    def __init__(self):
        self.in_waiting = 1
        self.time = 0
        print("MockSerial initialized")
    def readline(self):
        time.sleep(0.1)
        self.time += random.randint(50, 500)
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
