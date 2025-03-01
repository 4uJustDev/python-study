import random

array_size = 100
random_array = [random.randint(1, array_size) for _ in range(array_size)]
n = len(random_array)


def bubble_sort(arr):
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


bubble_sort(random_array)

print("Sorted array:", random_array)
