

def words_in_list(input_list: list[str]) -> int:
    """Compute the approximate number of words in a list of strings."""
    return sum(len(text.split(' ')) for text in input_list)


def combine_strings(input_list: list[str], word_limit: int, sep: str = '\n') -> list[str]:
    """Greedily combine a list of strings into a consolidated list of strings."""
    word_count = 0
    new_string = ''
    output = []
    for string in input_list:
        count = len(string.split(' '))
        if word_count+count<=word_limit:
            new_string += (sep + string)
            word_count += count
        else:
            if len(new_string)>0:
                output.append(new_string[1:])
                word_count = 0
                new_string = ''
            output.append(string)

    if len(new_string)>0:
        output.append(new_string[1:])

    return output