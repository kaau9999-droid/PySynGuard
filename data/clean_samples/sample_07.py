def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"


def sum_all(*args, **kwargs):
    total = sum(args)
    for key, value in kwargs.items():
        total += value
    return total


print(greet("Asaad"))
print(greet("Asaad", greeting="Hi"))
print(sum_all(1, 2, 3, bonus=10))
