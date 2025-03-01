import random

array_size = 100
random_array = [random.randint(1, array_size) for _ in range(array_size)]
n = len(random_array)


def gnome_sort(arr):
    n = len(arr)
    index = 0

    while index < n:
        if index == 0:
            index += 1
        elif arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1


gnome_sort(random_array)

print("Sorted array:", random_array)
