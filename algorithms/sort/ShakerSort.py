import random

array_size = 100
random_array = [random.randint(1, array_size) for _ in range(array_size)]
n = len(random_array)


def swap(arr, i):
    arr[i], arr[i + 1] = arr[i + 1], arr[i]


def shaker_sort(arr):
    is_swap = True
    start = 0
    end = n - 1

    while is_swap:
        is_swap = False

        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                swap(arr, i)
                is_swap = True

        if not is_swap:
            break

        is_swap = False
        end -= 1

        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                swap(arr, i)
                is_swap = True

        start += 1


shaker_sort(random_array)

print("Sorted array:", random_array)
