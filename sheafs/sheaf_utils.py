import random
import time

def berlekamp_massey(sequence):
    """
    Implements the Berlekamp-Massey algorithm to find the minimal LFSR
    polynomial for a given binary sequence.
    """
    n = len(sequence)
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1

    for i in range(n):
        d = int(sequence[i])
        for j in range(1, L + 1):
            d ^= C[j] * int(sequence[i - j])

        if d == 0:
            m += 1
        else:
            T = list(C)
            if len(C) < len(B) + m:
                C.extend([0] * (len(B) + m - len(C)))
            for j in range(len(B)):
                C[j + m] ^= B[j]

            if 2 * L <= i:
                L = i + 1 - L
                B = T
                b = d
                m = 1
            else:
                m += 1

    poly = 0
    for i, bit in enumerate(C):
        if bit: poly |= (1 << i)
    return poly

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
    def __init__(self, use_lfsr=True, noise_level=0.0):
        self.in_waiting = 1
        self.time = 0
        self.use_lfsr = use_lfsr
        self.noise_level = noise_level
        self.count = 0
        if self.use_lfsr:
            self.poly = 0x1D
            self.seed = random.randint(1, 255)
            self.gen = lfsr_gen(self.seed, self.poly)
        print(f"MockSerial initialized (LFSR={use_lfsr}, Noise={noise_level})")

    def readline(self):
        time.sleep(0.01)
        self.time += random.randint(50, 200)
        if self.use_lfsr:
            value = next(self.gen)
        else:
            value = random.randint(0, 1)

        if random.random() < self.noise_level:
            value ^= 1

        line = f"Collected Data Point: Time: {self.time} ms, Value: {value}\n"
        self.count += 1
        if self.count >= 32:
             line += "Data Collection Complete.\n"
             self.count = 0
        return line.encode('utf-8')

    def flushInput(self): pass
    def close(self): pass

def parse_data_line(line):
    if "Collected Data Point" in line:
        try:
            parts = line.split(',')
            time_part = parts[0].split('Time: ')[1].strip().replace(' ms', '')
            value_part = parts[1].split('Value: ')[1].strip()
            return int(time_part), int(value_part)
        except:
            pass
    return None, None
