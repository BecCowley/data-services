""" Use this snippet of code to decode binary flag codes into meaningful values
    Have to pass in flag labels or names as type list of strings.
    And i has to be an unsigned integer
    """


def convert(i, labels):
    output = ''
    pos = 0
    while i:
        if i & 1:
            output += labels[pos]
        pos += 1
        i >>= 1
    return output
