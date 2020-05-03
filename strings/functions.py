import functools
import itertools
import string

def is_palindromic(s):
    return all(s[i] == s[~i] for i in range(len(s) // 2))

def int_to_string(x):
    is_negative = False
    if x < 0:
        x, is_negative = -x, True

    s = []
    while True:
        s.append(chr(ord('0') + x % 10))
        x //= 10
        if x == 0:
            break

    return ('-' if is_negative else '') + ''.join(reversed(s))

def string_to_int(s):
    return functools.reduce(
        lambda running_sum, c: running_sum * 10 + string.digits.index(c),
        s[s[0] == '-':], 0) * (-1 if s[0] == '-' else 1)

def convert_base(num_as_string, b1, b2):
    def construct_from_base(num_as_int, base):
        return ('' if num_as_int == 0 else
                construct_from_base(num_as_int // base, base) +
                string.hexdigits[num_as_int % base].upper())

    is_negative = num_as_string[0] == '-'
    num_as_int = functools.reduce(
        lambda x, c: x * b1 + string.hexdigits.index(c.lower()),
        num_as_string[is_negative:], 0)
    return ('-' if is_negative else '') + ('0' if num_as_int == 0 else
                                           construct_from_base(num_as_int, b2))

def ss_decode_col_id(col):
    return functools.reduce(
        lambda result, c: result * 26 + ord(c) - ord('A') + 1, col, 0)

def replace_and_remove(size, s):
    write_idx, a_count = 0, 0
    for i in range(size):
        if s[i] != 'b':
            s[write_idx] = s[i]
            write_idx += 1
        if s[i] == 'a':
            a_count += 1

    cur_idx = write_idx - 1
    write_idx += a_count - 1
    final_size = write_idx + 1
    while cur_idx >= 0:
        if s[cur_idx] == 'a':
            s[write_idx - 1:write_idx + 1] = 'dd'
            write_idx -= 2
        else:
            s[write_idx] = s[cur_idx]
            write_idx -= 1
        cur_idx -= 1
    return final_size

def is_palindrome(s):
    i, j = 0, len(s) - 1
    while i < j:
        while not s[i].isalnum() and i < j:
            i += 1
        while not s[j].isalnum() and i < j:
            j -= 1
        if s[i].lower() != s[j].lower():
            return False
        i, j = i + 1, j - 1
    return True

def reverse_words(s):
    s.reverse()

    def reverse_range(s, start, end):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start, end = start + 1, end - 1

    start = 0
    while True:
        end = s.find(b' ', start)
        if end < 0:
            break
        reverse_range(s, start, end-1)
        start = end + 1
    reverse_range(s, start, len(s) - 1)

MAPPING = ('0','1','ABC','DEF','GHI','JKL','MNO','PQRS','TUV','WXYZ')

def phone_mnemonic(phone_number):
    def phone_mnemonic_helper(digit):
        if digit == len(phone_number):
            mnemonics.append(''.join(partial_mnemonic))
        else:
            for c in MAPPING[int(phone_number[digit])]:
                partial_mnemonic[digit] = c
                phone_mnemonic_helper(digit + 1)

    mnemonics, partial_mnemonic = [], [0] * len(phone_number)
    phone_mnemonic_helper(0)
    return mnemonics

def look_and_say(n):
    def next_number(s):
        result, i = [], 0
        while i < len(s):
            count = 1
            while i + 1 < len(s) and s[i] == s[i+1]:
                i += 1
                count += 1
            result.append(str(count) + s[i])
            i += 1
        return ''.join(result)
    s = '1'
    for _ in range(1, n):
        s = next_number(s)
    return s

def look_and_say_python(n):
    s = '1'
    for _ in range(n - 1):
        s = ''.join(
            str(len(list(group))) + key for key, group in itertools.groupby(s))
    return s

def roman_to_integer(s):
    T = {
        'I':1,
        'V':5,
        'X':10,
        'L':50,
        'C':100,
        'D':500,
        'M':1000
    }
    return functools.reduce(
        lambda val, i: val + (-T[s[i]] if T[s[i]] < T[s[i+1]] else T[s[i]]),
        reversed(range(len(s)-1)), T[s[-1]]
    )

def get_valid_ip_address(s):
    def is_valid_part(s):
        return len(s) == 1 or (s[0] != '0' and int(s) <= 255)

    result, parts = [], [None]*4
    for i in range(1, min(4, len(s))):
        parts[0] = s[:i]
        if is_valid_part(parts[0]):
            for j in range(1, min(len(s) - i, 4)):
                parts[1] = s[i:i + j]
                if is_valid_part(parts[1]):
                    for k in range(1, min(len(s) - i - j, 4)):
                        parts[2], parts[3] = s[i+j:i+j+k], s[i+j+k:]
                        if is_valid_part(parts[2]) and is_valid_part(parts[3]):
                            result.append('.'.join(parts))
    return result

def snake_string(s):
    result = []
    for i in range(1, len(s), 4):
        result.append(s[i])
    for i in range(0, len(s), 2):
        result.append(s[i])
    for i in range(3, len(s), 4):
        result.append(s[i])
    return ''.join(result)

def snake_string_pythonic(s):
    return s[1::4] + s[::2] + s[3::4]

def decoding(s):
    count, result = 0, []
    for c in s:
        if c.isdigit():
            count = count * 10 + int(c)
        else:
            result.append(c * count)
            count = 0
    return ''.join(result)

def encoding(s):
    result, count = [], 1
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[i-1]:
            result.append(str(count) + s[i-1])
            count = 1
        else:
            count += 1
    return ''.join(result)

def rabin_karp(t, s):
    if len(s) > len(t):
        return -1

    BASE = 26
    t_hash = functools.reduce(lambda h, c: h * BASE + ord(c), t[:len(s)], 0)
    s_hash = functools.reduce(lambda h, c: h * BASE + ord(c), s, 0)
    power_s = BASE**max(len(s) - 1, 0)
    for i in range(len(s), len(t)):
        if t_hash == s_hash and t[i - len(s):i] == s:
            return i - len(s)
        t_hash -= ord(t[i - len(s)]) * power_s
        t_hash = t_hash * BASE + ord(t[i])
    if t_hash == s_hash and t[-len(s):] == s:
        return len(t) - len(s)
    return -1
