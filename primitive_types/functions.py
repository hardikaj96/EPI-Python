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
