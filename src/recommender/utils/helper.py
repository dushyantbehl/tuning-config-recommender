def set_difference(l1, l2):
    # l1 - l2
    diff = []
    for d1 in l1:
        if d1 not in l2:
            diff.append(d1)
    return diff


def set_issubset(l1, l2):
    # if l2 is subset of l1
    issubset = True
    for item in l2:
        if item not in l1:
            return False
    return issubset
