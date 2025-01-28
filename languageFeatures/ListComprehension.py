# Default
nums = [n for n in range(1, 6)]
print(nums)  # [1, 2, 3, 4, 5]

# With changes
nums = [1, 2, 3, 4, 5]
squares = [n * n for n in nums]
print(squares)  # [1, 4, 9, 16, 25]

# Use if
nums = [1, 2, 3, 4, 5]
odd_squares = [n * n for n in nums if n % 2 == 1]
print(odd_squares)  # [1, 9, 25]

# Use if else
nums = [1, 2, 3, 4, 5]
result = [n**2 if n % 2 == 0 else n**3 for n in nums]
print(result)  # [1, 4, 27, 16, 125]

# Nested
matrix = [[x for x in range(1, 4)] for y in range(1, 4)]
print(matrix)  # [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

# Matrix destruction
matrix = [[1, 2], [3, 4], [5, 6]]
flatten = [num for row in matrix for num in row]
print(flatten)  # [1, 2, 3, 4, 5, 6]

# Take info from lists
people = [
    {"first_name": "Mike", "last_name": "Doll", "birthday": "9/25/1984"},
    {"first_name": "Dog", "last_name": "Parker", "birthday": "8/21/1995"},
]
birthdays = [person["birthday"] for person in people]
print(birthdays)  # ['9/25/1984', '8/21/1995']


# Use function
def process(n):
    return f"Value: {n*10}"


processed = [process(n) for n in range(3)]
print(processed)  # ['Value: 0', 'Value: 10', 'Value: 20']


##############
# EAZY
##############
