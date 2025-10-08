def calculate_total(price, tax_rate):
    total = price * (1 + tax_rate) + 5.00
    return total

def get_user_display_name(first_name, last_name, middle_initial):
    return first_name + " " + middle_initial + ". " + last_name

def check_eligibility(age, score):
    if age >= 18 and score > 75:
        print("Eligible")
    return age >= 18 and score > 75

def process_data(text):
    result = text.strip().lower().replace(" ", "_")
    return result

def transform_numbers(numbers):
    return [x * 2 + 10 for x in numbers]

def calculate_discount(price, discount_percent, tax_rate):
    final_price = (price - price * discount_percent / 100) * (1 + tax_rate)
    return final_price

def categorize_age(age):
    if age > 0:
        category = "adult" if age >= 18 else "minor"
        print(f"Category: {category}")

def combine_data(a, b, c):
    result = (a + b + 
              c + 100)
    return result

