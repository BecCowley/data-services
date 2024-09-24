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


# test it:
# val = 2**2 + 2**7
# labels = ['No_QC_performed', 'Good_data', 'Probably_good_data', 'Bad_data_that_are_potentially_correctable', 'Bad_data', 'Value_changed', 'Not_used', 'Not_used', 'Not_used', 'Missing_value']
# print(convert(val, labels))

