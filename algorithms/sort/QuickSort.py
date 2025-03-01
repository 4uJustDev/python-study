import random

array_size = 100
random_array = [random.randint(1, array_size) for _ in range(array_size)]
n = len(random_array)


def quick_sort(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)

        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)


def partition(arr, low, high):

    pivot = arr[high]

    i = low - 1

    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return i + 1


quick_sort(random_array, 0, len(random_array) - 1)

print("Sorted array:", random_array)
