import random

array_size = 100
random_array = [random.randint(1, array_size) for _ in range(array_size)]
n = len(random_array)


def insertion_sort(arr):
    for i in range(1, n):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

insertion_sort(random_array)

print("Sorted array:", random_array)
