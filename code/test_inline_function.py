def add(a, b):
    return a + b

result1 = add(5, 3)
result2 = add(10, 20)


def calculate_tax(price, rate):
    return price * rate

total_tax = calculate_tax(100, 0.08)


def full_name(first, last):
    return first + " " + last

name = full_name("John", "Doe")
another_name = full_name("Jane", "Smith")


def discount(price, percent):
    return price * (1 - percent / 100)

discounted = discount(price=50, percent=10)


def format_message(name, greeting):
    return greeting + ", " + name + "!"

msg = format_message("Alice", greeting="Hello")


def double(x):
    return x * 2

def triple(x):
    return x * 3

result = double(triple(5))


def calculate_area(width, height):
    return width * height * 3.14159

area = calculate_area(10, 20)


def normalize(text):
    return text.strip().lower()

normalized = normalize("  HELLO WORLD  ")


def square(n):
    return n * n

final = square(4) + square(5)


def is_adult(age):
    return age >= 18

if is_adult(25):
    print("Adult")

