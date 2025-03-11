
# Define a function to categorize vehicle types based on car names
def categorize_vehicle_type(car_name):
    if 'wagon' in car_name.lower():
        return 'wagon'
    elif 'sedan' in car_name.lower():
        return 'sedan'
    elif 'convertible' in car_name.lower():
        return 'convertible'
    elif 'coupe' in car_name.lower():
        return 'coupe'
    elif 'hatchback' in car_name.lower():
        return 'hatchback'
    elif 'pickup' in car_name.lower():
        return 'pickup'
    else:
        return 'other'