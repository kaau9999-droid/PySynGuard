def timer(func):
    def wrapper(*args, **kwargs):
        print("Starting...")
        result = func(*args, **kwargs)
        print("Finished.")
        return result
    return wrapper


@timer
def compute_square(n):
    return n ** 2


print(compute_square(6))
