import random
from datetime import datetime

def shuffle_multiple_lists(*lists):
    # Combine the lists into tuples
    combined_lists = list(zip(*lists))

    # Shuffle the combined list
    random.shuffle(combined_lists)

    # Unzip the shuffled list back into separate lists
    shuffled_lists = zip(*combined_lists)

    # Convert the result back into separate lists
    result_lists = [list(lst) for lst in shuffled_lists]

    return result_lists


def map_labels(labels):
    d = dict()
    counter = 1
    for label in labels:
        if label not in d:
            d[label] = counter
            counter += 1
    return d


def generate_model_name():
    datetime_str = str(datetime.now())
    datetime_str = datetime_str.split(":")
    datetime_str.pop()
    datetime_str = ":".join(datetime_str)
    datetime_str = datetime_str.replace(" ", "_").replace("-", "_")
    return f"model_{datetime_str}"
