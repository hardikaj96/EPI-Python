import math
from random import randint
import collections
# The time complexity O(n)
# n is the word size
def parity(x):
    result = 0
    while x:
        result ^= x & 1
        x >>= 1
    return result

# The time complexity O(k)
# k is the number of bits set to 1
def parity1(x):
    result = 0
    while x:
        result ^= 1
        x &= x - 1
    return result

# The time complexity O(logn)
# n is the word size
def parity2(x):
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 0x1


def swap_bits(x, i, j):
    if (x >> i) & 1 != (x >> j) & 1:
        bit_mask = (1 << i) | (1 << j)
        x ^= bit_mask
    return x

def closest_int_same_bit_count(x):
    NUM_UNSIGNED_BITS = 64
    for i in range(NUM_UNSIGNED_BITS-1):
        if (x >> i) & 1 != (x >> (i +1)) & 1:
            x ^= (1 << i) | (1 << (i + 1))
            return x
    raise ValueError('All bits are 0 or 1')

# Time complexity O(n^2)
def multiple(x, y):
    def add(a, b):
        running_sum, carryin, k, temp_a, temp_b = 0, 0, 1, a, b
        while temp_a or temp_b:
            ak, bk = a & k, b & k
            carryout = (ak & bk) | (ak & carryin) | (bk & carryin)
            running_sum |= ak ^ bk ^ carryin
            carryin, k, temp_a, temp_b = (carryout << 1, k << 1,
                                          temp_a >> 1, temp_b >> 1)
        return running_sum | carryin

    running_sum = 0
    while x:
        if x & 1:
            running_sum = add(running_sum, y)
        x, y = x >> 1, y << 1
    return running_sum

# Time complexity O(n)
def divide(x, y):
    result, power = 0, 32
    y_power = y << power
    while x >= y:
        while y_power > x:
            y_power >>= 1
            power -= 1
        result += 1 << power
        x -= y_power
    return result


# Time complexity O(n)
def power(x, y):
    result, power = 1.0, y
    if y < 0:
        power, x = -power, 1.0 / x
    while power:
        if power & 1:
            result *= x
        x, power = x * x, power >> 1
    return result

def reverse(x):
    result, x_remaining = 0,  abs(x)
    while x_remaining:
        result = result * 10 + x_remaining % 10
        x_remaining //= 10
    return -result if x<0 else result

def is_palindrome_number(x):
    if x <= 0:
        return x == 0
    num_digits = math.floor(math.log10(x)) + 1
    msd_mask = 10**(num_digits-1)
    for i in range(num_digits // 2):
        if x // msd_mask != x % 10:
            return False
        x %= msd_mask
        x //= 10
        msd_mask //= 100
    return True

def uniform_random(lower_bound, upper_bound):
    number_of_outcomes = upper_bound - lower_bound + 1
    while True:
        result, i = 0, 0
        while (1 << i) < number_of_outcomes:
            result = (result << 1) | randint(0, 2)
            i += 1
        if result < number_of_outcomes:
            break
    return result + lower_bound

Rectangle = collections.namedtuple('Rectangle', ('x', 'y', 'width', 'height'))
def intersect_rectangle(R1, R2):
    def is_intersect(R1, R2):
        return (R1.x <= R2.x + R2.width and R1.x + R1.width >= R2.x
                and R1.y <= R2.y + R2.height and R1.y + R1.height >= R2.y)
    if not is_intersect(R1, R2):
        return Rectangle(0, 0, -1, -1)
    return Rectangle(
        max(R1.x, R2.x),
        max(R1.y, R2.y),
        min(R1.x + R1.width, R2.x + R2.width) - max(R1.x, R2.x),
        min(R1.y + R1.height, R2.y + R2.height) - max(R1.y, R2.y),
    )