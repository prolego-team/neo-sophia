import neosophia.text_utils as tu


def test_get_capitalized_phrases():
    assert tu.get_capitalized_phrases('This is a Test Case.')==['Test Case']
    assert tu.get_capitalized_phrases('This is a Test Case for sure.')==['Test Case']
    assert tu.get_capitalized_phrases('This Is a Test Case.')==['This Is', 'Test Case']
    assert tu.get_capitalized_phrases('This Is a Test Case?')==['This Is', 'Test Case']
    assert tu.get_capitalized_phrases('This sure Is  A  big Test Case  here.')==['Is A', 'Test Case']
    assert tu.get_capitalized_phrases('This')==[]
