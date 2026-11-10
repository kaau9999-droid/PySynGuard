def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    else:
        return result
    finally:
        print("Division attempted")


print(safe_divide(10, 2))
print(safe_divide(10, 0))
