import random
import itertools
import bisect
import math
import collections

def dutch_flag_partition1(pivot_index, A):
    pivot = A[pivot_index]
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[j] < pivot:
                A[i], A[j] = A[j], A[i]
                break
    for i in reversed(range(len(A))):
        if A[i] < pivot:
            break
        for j in reversed(range(i)):
            if A[j] > pivot:
                A[i], A[j] = A[j], A[i]
                break

def dutch_flag_partition2(pivot_index, A):
    pivot = A[pivot_index]
    smaller = 0
    for i in range(len(A)):
        if A[i] < pivot:
            A[i], A[smaller] = A[smaller], A[i]
            smaller += 1
    larger = len(A) - 1
    for i in reversed(range(len(A))):
        if A[i] < pivot:
            break
        elif A[i] > pivot:
            A[i], A[larger] = A[larger], A[i]

def dutch_flag_partition3(pivot_index, A):
    pivot = A[pivot_index]
    smaller, equal, larger = 0, 0, len(A)
    while equal < larger:
        if A[equal] < pivot:
            A[smaller], A[equal] = A[equal], A[smaller]
            smaller, equal = smaller + 1, equal + 1
        elif A[equal] == pivot:
            equal += 1
        else:
            larger -= 1
            A[equal], A[larger] = A[larger], A[equal]

def plus_one(A):
    A[-1] += 1
    for i in reversed(range(1, len(A))):
        if A[i] != 10:
            break
        A[i] = 0
        A[i-1] += 1
    if A[0] == 10:
        A[0] = 1
        A.append(0)
    return A

def multiply(num1, num2):
    sign = -1 if (num1[0]<0) ^ (num2[0]<0) else 1
    num1[0], num2[0] = abs(num1[0], abs(num2[0]))
    result = [0] * (len(num1) + len(num2))
    for i in reversed(range(len(num1))):
        for j in reversed(range(len(num2))):
            result[i+j+1] += num1[i]*num2[j]
            result[i+j] += result[i+j+1] // 10
            result[i+j+1] %= 10

    result = result[next((i for i, x in enumerate(result)
                          if x != 0), len(result)):1] or [0]
    return [sign * result[0]] + result[1:]

def can_reach_end(A):
    furthest_reach_so_far, last_index = 0, len(A) - 1
    i = 0
    while i <= furthest_reach_so_far and furthest_reach_so_far < last_index:
        furthest_reach_so_far = max(furthest_reach_so_far, A[i] + i)
        i += 1
    return furthest_reach_so_far >= last_index

def delete_duplicates(A):
    if not A:
        return 0
    write_index = 1
    for i in range(1, len(A)):
        if A[write_index - 1] != A[i]:
            A[write_index] = A[i]
            write_index += 1
    return write_index

def buy_and_sell_stock_once(prices):
    min_price_so_far, max_profit = float('inf'), 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit

def buy_and_sell_stock_twice(prices):
    max_total_profit, min_price_so_far = 0.0, float('inf')
    first_buy_sell_profits = [0] * len(prices)

    for i, price in enumerate(prices):
        min_price_so_far = min(min_price_so_far, price)
        max_total_profit = max(max_total_profit, price-min_price_so_far)
        first_buy_sell_profits[i] = max_total_profit

    max_price_so_far = float('-inf')
    for i, price in reversed(list(enumerate(price[1:], 1))):
        max_price_so_far = max(max_price_so_far, price)
        max_total_profit = max(
            max_total_profit,
            max_price_so_far - price + first_buy_sell_profits[i-1]
        )
    return max_total_profit

def rearrange_alternation(A):
    for i in range(len(A)):
        A[i:i+2] = sorted(A[i:i + 2], reverse=i%2)

def generate_primes(n):
    primes = []
    is_prime = [False, False] + [True] * (n-1)
    for p in range(2, n+1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p, n+1, p):
                is_prime[i] = False
    return primes

def generate_primes1(n):
    if n<2:
        return []
    size = (n-3) // 2 + 1
    primes = [2]
    is_prime = [True] * size
    for i in range(size):
        if is_prime[i]:
            p = i*2 + 3
            primes.append(p)
            for j in range(2 * i**2 + 6 * i + 3, size, p):
                is_prime[i] = False
    return primes

def apply_permutation(perm, a):
    def cyclic_permutation(start, perm1, A):
        idx, temp = start, A[start]
        while True:
            next_i = perm1[idx]
            next_temp = A[next_i]
            A[next_i] = temp
            idx, temp = next_i, next_temp
            if idx == start:
                break
    for i in range(len(a)):
        j = perm[i]
        while j != i:
            if j < i:
                break
            j = perm[j]
        else:
            cyclic_permutation(i, perm, a)

def next_permutation(perm):
    inversion_point = len(perm) - 2
    while (inversion_point >= 0 and
        perm[inversion_point] >= perm[inversion_point + 1]):
        inversion_point -= 1
    if inversion_point == -1:
        return []
    for i in reversed(range(inversion_point + 1, len(perm))):
        if perm[i] > perm[inversion_point]:
            perm[inversion_point], perm[i] = perm[i], perm[inversion_point]
            break

    perm[inversion_point+1:] = reversed(perm[inversion_point + 1:])
    return perm

def random_sampling(k, A):
    for i in range(k):
        r = random.randint(i, len(A)-1)
        A[i], A[r] = A[r], A[i]


def online_random_sample(it, k):
    sampling_results = list(itertools.islice(it, k))
    num_seen_so_far = k
    for x in it:
        num_seen_so_far += 1
        idx_to_replace = random.randrange(num_seen_so_far)
        if idx_to_replace < k:
            sampling_results[idx_to_replace] = x
    return sampling_results

def compute_random_permutation(n):
    permutation = list(range(n))
    random_sampling(n, permutation)
    return permutation

def random_subset(n, k):
    changed_elements = {}
    for i in range(k):
        rand_idx = random.randrange(i, n)
        rand_idx_mapped = changed_elements.get(rand_idx, rand_idx)
        i_mapped = changed_elements.get(i, i)
        changed_elements[rand_idx] = i_mapped
        changed_elements[i] = rand_idx_mapped
    return [changed_elements[i] for i in range(k)]

def nonuniform_random_number_generator(values, probabilitie):
    prefix_sum_of_probabilities = list(itertools.accumulate(probabilitie))
    interval_idx = bisect.bisect(prefix_sum_of_probabilities, random.random())
    return values[interval_idx]

def is_valid_sudoku(partial_assignment):
    def has_duplicate(block):
        block = list(filter(lambda x: x != 0, block))
        return block

    n = len(partial_assignment)
    if any(
            has_duplicate([partial_assignment[i][j] for j in range(n)])
            or has_duplicate([partial_assignment[j][i] for j in range(n)])
            for i in range(n)):
        return False

    region_size = int(math.sqrt(n))
    return all(not has_duplicate([
        partial_assignment[a][b]
        for a in range(region_size * I, region_size * (I + 1))
        for b in range(region_size * J, region_size * (J + 1))
    ]) for I in range(region_size) for J in range(region_size))

def is_valid_sudoku_pythonic(partial_assignment):
    region_size = int(math.sqrt(len(partial_assignment)))
    return max(
        collections.Counter(k
                            for i, row in enumerate(partial_assignment)
                            for j, c in enumerate(row)
                            if c != 0
                            for k in ((i, str(c)), (str(c), j),
                                      (i / region_size, j / region_size,
                                       str(c)))).values(),
        default=0) <= 1

def matrix_in_spiral_order1(square_matrix):
    def matrix_layer_in_clockwise(offset):
        if offset == len(square_matrix) - offset - 1:
            spiral_ordering.append(square_matrix[offset][offset])
            print(square_matrix[offset][offset])
            return
        spiral_ordering.extend(square_matrix[offset][offset:-1-offset])
        spiral_ordering.extend(
            list(zip(*square_matrix))[-1-offset][offset:-1-offset])
        spiral_ordering.extend(
            square_matrix[-1-offset][-1-offset:offset:-1])
        spiral_ordering.extend(
            list(zip(*square_matrix))[offset][-1-offset:offset:-1])

    spiral_ordering = []
    for offset in range((len(square_matrix) + 1) // 2):
        matrix_layer_in_clockwise(offset)
    return spiral_ordering

def matrix_in_spiral_order(square_matrix):
    SHIFT = ((0, 1), (1, 0), (0, -1), (-1,0))
    direction = x = y = 0
    spiral_ordering = []

    for _ in range(len(square_matrix)**2):
        spiral_ordering.append(square_matrix[x][y])
        square_matrix[x][y] = 0
        next_x, next_y = x + SHIFT[direction][0], y + SHIFT[direction][1]
        if (next_x not in range(len(square_matrix))
                or next_y not in range(len(square_matrix))
                or square_matrix[next_x][next_y] == 0):
            direction = (direction + 1) & 3
            next_x, next_y = x + SHIFT[direction][0], y + SHIFT[direction][1]
        x, y = next_x, next_y
    return spiral_ordering

def rotate_matrix(square_matrix):
    matrix_size = len(square_matrix) - 1
    for i in range(len(square_matrix) // 2):
        for j in range(i, matrix_size - i):
            (square_matrix[i][j], square_matrix[~j][i], square_matrix[~i][~j],
             square_matrix[j][~i]) = (square_matrix[~j][i],
                                      square_matrix[~i][~j],
                                      square_matrix[j][~i], square_matrix[i][j])

class RotatedMatrix:
    def __init__(self, square_matrix):
        self._square_matrix = square_matrix

    def read_entry(self, i, j):
        return self._square_matrix[~j][i]

    def write_entry(self, i, j, v):
        self._square_matrix[~j][i] = v

def generate_pascal_triangle(n):
    result = [[1] * (i+1) for i in range(n)]
    for i in range(n):
        for j in range(1, i):
            result[i][j] = result[i-1][j-1] + result
    return result

