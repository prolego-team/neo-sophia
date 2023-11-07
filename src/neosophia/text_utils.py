"""Generic text processing utilities."""

import re

def words_in_list(input_list: list[str]) -> int:
    """Compute the approximate number of words in a list of strings."""
    return sum(len(text.strip().split(' ')) for text in input_list)


def combine_strings(input_list: list[str], word_limit: int, sep: str = '\n') -> list[str]:
    """Greedily combine a list of strings into a consolidated list of strings."""
    word_count = 0
    new_string = ''
    output = []
    for string in input_list:
        string_trimmed = string.strip()
        count = len(string_trimmed.split(' '))
        if word_count+count<=word_limit:
            new_string += (sep + string_trimmed)
            word_count += count
        else:
            if len(new_string)>0:
                output.append(new_string[1:])
                word_count = count
                new_string = sep + string_trimmed

    if len(new_string)>0:
        output.append(new_string[1:])

    return output


def get_capitalized_phrases(text: str) -> list[str]:
    """Extract capitalized phrases from a string."""
    capped_phrases = []
    mid_phrase = False
    text = re.sub(r'[.?!()]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.strip().split(' ')
    new_phrase = []
    for word in words:
        is_cap = word[0].isupper()
        if is_cap:
            new_phrase.append(word)
            mid_phrase = True
        elif mid_phrase and (not is_cap):
            capped_phrases.append(new_phrase)
            mid_phrase = False
            new_phrase = []
        else: # not mid_phrase and not is_cap
            pass

    if len(new_phrase)>0:
        capped_phrases.append(new_phrase)

    if len(capped_phrases)>0:
        if len(capped_phrases[0])==1 and capped_phrases[0][0]==words[0]:
            capped_phrases.pop(0)
    return [' '.join(phrase) for phrase in capped_phrases]
