import random


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


# Example usage:
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c', 'd', 'e']
list3 = ['apple', 'banana', 'orange', 'grape', 'pear']

# Shuffle the lists together
shuffled_list1, shuffled_list2, shuffled_list3 = shuffle_multiple_lists(list1, list2, list3)

# Print the shuffled lists
print("Shuffled list1:", shuffled_list1)
print("Shuffled list2:", shuffled_list2)
print("Shuffled list3:", shuffled_list3)
