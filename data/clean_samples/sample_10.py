import math
import json


def circle_stats(radius):
    area = math.pi * radius ** 2
    circumference = 2 * math.pi * radius
    return {
        "radius": radius,
        "area": round(area, 2),
        "circumference": round(circumference, 2),
    }


stats = circle_stats(5)
print(json.dumps(stats, indent=2))
