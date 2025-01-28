import timeit
from tqdm import tqdm


# Simple loops
def test_for():
    for _ in range(10**6):
        pass


def test_while():
    i = 0
    while i < 10**6:
        i += 1


# Loops + math
def test_for_math():
    total = 0
    for i in range(10**6):
        total += i * 2


def test_while_math():
    total = 0
    i = 0
    while i < 10**6:
        total += i * 2
        i += 1


# Lists
def test_for_list():
    lst = [i for i in range(10**5)]
    for item in lst:
        pass


def test_while_list():
    lst = [i for i in range(10**5)]
    i = 0
    while i < len(lst):
        item = lst[i]
        i += 1


# Nested
def test_nested_for():
    for i in range(1000):
        for j in range(1000):
            pass


def test_nested_while():
    i = 0
    while i < 1000:
        j = 0
        while j < 1000:
            j += 1
        i += 1


# Break test
def test_for_break():
    for i in range(10**6):
        if i >= 500000:
            break


def test_while_break():
    i = 0
    while True:
        if i >= 500000:
            break
        i += 1


test_pairs = [
    ("For VS While (simple)", test_for, test_while),
    ("For VS While (math)", test_for_math, test_while_math),
    ("For VS While (list)", test_for_list, test_while_list),
    ("Nested VS While (nested)", test_nested_for, test_nested_while),
    ("For VS While (break)", test_for_break, test_while_break),
]

results = {}
with tqdm(
    total=len(test_pairs),
    desc="🚀 Performance tests",
    unit="pair",
    ncols=80,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
) as pbar:
    for title, for_func, while_func in test_pairs:
        # run tests
        for_time = timeit.timeit(for_func, number=100)
        while_time = timeit.timeit(while_func, number=100)

        # Determine winner
        winner = None
        if for_time < while_time:
            diff = while_time - for_time
            winner_str = f"FOR (+{diff:.4f}s)"
        else:
            diff = for_time - while_time
            winner_str = f"WHILE (+{diff:.4f}s)"

        # Output
        tqdm.write(
            f"{title}:\n"
            f"  FOR: {for_time:.4f}s | WHILE: {while_time:.4f}s\n"
            f"  Winner: {winner_str}\n"
            f"{'-'*40}"
        )

        pbar.update(1)
