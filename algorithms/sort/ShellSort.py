import random

array_size = 100
random_array = [random.randint(1, array_size) for _ in range(array_size)]
n = len(random_array)


def shell_sort(arr):
    n = len(arr)

    gap = 1
    while gap < n // 3:
        gap = gap * 3 + 1

    while gap >= 1:
        for i in range(gap, n):
            temp = arr[i]
            j = i

            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap = gap // 3


shell_sort(random_array)


print("Sorted array:", random_array)
