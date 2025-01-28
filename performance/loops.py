import timeit


def test_for():
    for _ in range(10**6):
        pass


def test_while():
    i = 0
    while i < 10**6:
        i += 1


for_time = timeit.timeit(test_for, number=10)
while_time = timeit.timeit(test_while, number=10)

print(f"For loop (avg): {for_time / 10:.5f} s")
print(f"While loop (avg): {while_time / 10:.5f} s")
