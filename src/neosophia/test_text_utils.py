from neosophia.text_utils import words_in_list, get_capitalized_phrases, combine_strings


def test_words_in_list():
    inp = [' This is', 'a test.\n ']
    assert words_in_list(inp)==4

    inp = ['This is a test.']
    assert words_in_list(inp)==4

    inp = []
    assert words_in_list(inp)==0


def test_combine_strings():
    inp = ['This', 'is', 'a', 'test']
    assert combine_strings(inp, 1)==['This', 'is', 'a', 'test']
    assert combine_strings(inp, 2)==['This is', 'a test']
    assert combine_strings(inp, 3)==['This is a', 'test']
    assert combine_strings(inp, 4)==['This is a test']
    assert combine_strings(inp, 5)==['This is a test']


def test_get_capitalized_phrases():
    assert get_capitalized_phrases('This is a Test Case.')==['Test Case']
    assert get_capitalized_phrases('This is a Test Case for sure.')==['Test Case']
    assert get_capitalized_phrases('This Is a Test Case.')==['This Is', 'Test Case']
    assert get_capitalized_phrases('This Is a Test Case?')==['This Is', 'Test Case']
    assert get_capitalized_phrases('This sure Is  A  big Test Case  here.')==['Is A', 'Test Case']
    assert get_capitalized_phrases('This')==[]
