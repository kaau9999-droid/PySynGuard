student_scores = {"Ali": 90, "Sara": 85, "Omar": 78}

squares = [x * x for x in range(10)]
passed = {name: score for name, score in student_scores.items() if score >= 80}

for name, score in passed.items():
    print(f"{name}: {score}")

print(squares)
