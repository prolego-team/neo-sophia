from neosophia.search.utils.doctree import (
    Section,
    parse,
    search,
    flatten_doctree,
    get_supersection,
    get_subsection,
    expand,
    get_tree_text,
    consolidate_leaves,
    consolidate_paragraphs
)

def example():
    flat_input = [
        (1, Section('Introduction', ['Paragraph 1', 'Paragraph 2'])),
        (1, Section('1. Section One', ['Paragraph 3', 'Paragraph 4'])),
        (2, Section('1.1 Subsection A', ['Paragraph 5', 'Paragraph 6'])),
        (2, Section('1.2 Subsection B', ['Paragraph 7', 'Paragraph 8'])),
        (1, Section('2. Section Two', ['Paragraph 9', 'Paragraph 10'])),
        (2, Section('2.1 Subsection A', ['Paragraph 11', 'Paragraph 12'])),
        (3, Section('2.1.1 Fine details', ['Detailed information', 'Even more details'])),
        (3, Section('2.1.2 Extra details', ['Fine as frog hair', 'Fine as espresso grounds'])),
        (2, Section('2.2 Subsection B', ['Paragraph 13', 'Paragraph 14'])),
        (1, Section('3. Section Three', ['Paragraph 15', 'Paragraph 16']))
    ]
    example_doctree = [
        Section('Introduction', ['Paragraph 1', 'Paragraph 2']),
        Section('1. Section One', ['Paragraph 3', 'Paragraph 4']),
        [
            Section('1.1 Subsection A', ['Paragraph 5', 'Paragraph 6']),
            Section('1.2 Subsection B', ['Paragraph 7', 'Paragraph 8'])
        ],
        Section('2. Section Two', ['Paragraph 9', 'Paragraph 10']),
        [
            Section('2.1 Subsection A', ['Paragraph 11', 'Paragraph 12']),
            [
                Section('2.1.1 Fine details', ['Detailed information', 'Even more details']),
                Section('2.1.2 Extra details', ['Fine as frog hair', 'Fine as espresso grounds'])
            ],
            Section('2.2 Subsection B', ['Paragraph 13', 'Paragraph 14'])
        ],
        Section('3. Section Three', ['Paragraph 15', 'Paragraph 16'])
    ]
    return flat_input, example_doctree


def test_parse():
    flat_input, example_doctree = example()
    levels, sections = zip(*flat_input)
    assert parse(sections, levels)==example_doctree


def test_search():
    _, example_doctree = example()

    ind = search(example_doctree, lambda inp: '2.' in inp.title)
    assert len(ind)==5
    assert ind==[(3,), (4,0), (4,1,0), (4,1,1), (4,2)]

    ind = search(example_doctree, lambda inp: '1.1 Subsection A' in inp.title)
    assert len(ind)==1
    assert ind==[(2,0)]

    ind = search(example_doctree, lambda inp: 'Not there' in inp.title)
    assert len(ind)==0


def test_flatten_doctree():
    _, example_doctree = example()
    target = [
        # ((0,), 'Introduction'),
        ((0,0),  'Paragraph 1'),
        ((0,1),  'Paragraph 2'),
        # ((1,), '1. Section One'),
        ((1,0),  'Paragraph 3'),
        ((1,1),  'Paragraph 4'),
        # ((2,), '1.1 Subsection A'),
        ((2,0,0),  'Paragraph 5'),
        ((2,0,1),  'Paragraph 6'),
        # ((2,), '1.2 Subsection B'),
        ((2,1,0),  'Paragraph 7'),
        ((2,1,1),  'Paragraph 8'),
        # ((3,), '2. Section Two'),
        ((3,0),  'Paragraph 9'),
        ((3,1),  'Paragraph 10'),
        # ((4,), '2.1 Subsection A'),
        ((4,0,0),  'Paragraph 11'),
        ((4,0,1),  'Paragraph 12'),
        # ((5,), '2.1.1 Fine details'),
        ((4,1,0,0),  'Detailed information'),
        ((4,1,0,1),  'Even more details'),
        # ((5,), '2.1.2 Extra details'),
        ((4,1,1,0),  'Fine as frog hair'),
        ((4,1,1,1),  'Fine as espresso grounds'),
        # ((6,), '2.2 Subsection B'),
        ((4,2,0),  'Paragraph 13'),
        ((4,2,1),  'Paragraph 14'),
        # ((7,), '3. Section Three'),
        ((5,0),  'Paragraph 15'),
        ((5,1),  'Paragraph 16')
    ]
    flat = list(flatten_doctree(example_doctree))
    assert flat==target


def test_get_supersection():
    _, example_doctree = example()

    assert get_supersection(example_doctree, (4,0))=='Paragraph 10'
    assert get_supersection(example_doctree, (4,1,1))=='Paragraph 12'
    assert get_supersection(example_doctree, (0,))==None
    assert get_supersection(example_doctree, (5,))==None


def test_get_subsection():
    _, example_doctree = example()

    assert get_subsection(example_doctree, (4,0))=='Detailed information'
    assert get_subsection(example_doctree, (1,))=='Paragraph 5'


def test_expand():
    _, example_doctree = example()

    # only super section
    item, ind = 'Paragraph 13', (4,2)
    assert expand(item, example_doctree, ind)==[
        item,
        'Paragraph 10 '+item,
    ]

    # only sub section
    item, ind = 'Paragraph 4', (1,)
    assert expand(item, example_doctree, ind)==[
        item,
        item+' Paragraph 5',
    ]

    # both super and sub sections
    item, ind = 'Paragraph 11', (4,0)
    assert expand(item, example_doctree, ind)==[
        item,
        'Paragraph 10 '+item,
        item+' Detailed information',
        'Paragraph 10 '+item+' Detailed information',
    ]

    # neither super nor sub sections
    item, ind = 'Paragraph 1', (0,)
    assert expand(item, example_doctree, ind)==[
        item,
    ]


def test_get_tree_text():
    _, example_doctree = example()

    # get all of a section
    assert get_tree_text(example_doctree[3:5])==(
        '2. Section Two\nParagraph 9\nParagraph 10\n'
        '2.1 Subsection A\nParagraph 11\nParagraph 12\n'
        '2.1.1 Fine details\nDetailed information\nEven more details\n'
        '2.1.2 Extra details\nFine as frog hair\nFine as espresso grounds\n'
        '2.2 Subsection B\nParagraph 13\nParagraph 14\n'
    )
    assert get_tree_text(example_doctree[3:5], start_level=1)==(
        '2.1 Subsection A\nParagraph 11\nParagraph 12\n'
        '2.1.1 Fine details\nDetailed information\nEven more details\n'
        '2.1.2 Extra details\nFine as frog hair\nFine as espresso grounds\n'
        '2.2 Subsection B\nParagraph 13\nParagraph 14\n'
    )
    assert get_tree_text(example_doctree[3:5], end_level=1)==(
        '2. Section Two\nParagraph 9\nParagraph 10\n'
        '2.1 Subsection A\nParagraph 11\nParagraph 12\n'
        '2.2 Subsection B\nParagraph 13\nParagraph 14\n'
    )
    assert get_tree_text(example_doctree[3:5], start_level=1, end_level=1)==(
        '2.1 Subsection A\nParagraph 11\nParagraph 12\n'
        '2.2 Subsection B\nParagraph 13\nParagraph 14\n'
    )


def test_consolidate_leaves():
    _, example_doctree = example()
    assert consolidate_leaves(example_doctree, word_thresh=20)==[
        Section('Introduction', ['Paragraph 1', 'Paragraph 2']),
        Section('1. Section One', ['Paragraph 3', 'Paragraph 4']),
        [
            Section(
                '1.1 Subsection A-1.2 Subsection B',
                [
                    'Paragraph 5',
                    'Paragraph 6',
                    'Paragraph 7',
                    'Paragraph 8'
                ]
            ),
        ],
        Section('2. Section Two', ['Paragraph 9', 'Paragraph 10']),
        [
            Section('2.1 Subsection A', ['Paragraph 11', 'Paragraph 12']),
            [
                Section(
                    '2.1.1 Fine details-2.1.2 Extra details',
                    [
                        'Detailed information',
                        'Even more details',
                        'Fine as frog hair',
                        'Fine as espresso grounds'
                    ]
                )
            ],
            Section('2.2 Subsection B', ['Paragraph 13', 'Paragraph 14'])
        ],
        Section('3. Section Three', ['Paragraph 15', 'Paragraph 16'])
    ]


def test_consolidate_paragraphs():
    _, example_doctree = example()
    assert consolidate_paragraphs(example_doctree, word_limit=4)==[
        Section('Introduction', ['Paragraph 1 Paragraph 2']),
        Section('1. Section One', ['Paragraph 3 Paragraph 4']),
        [
            Section('1.1 Subsection A', ['Paragraph 5 Paragraph 6']),
            Section('1.2 Subsection B', ['Paragraph 7 Paragraph 8'])
        ],
        Section('2. Section Two', ['Paragraph 9 Paragraph 10']),
        [
            Section('2.1 Subsection A', ['Paragraph 11 Paragraph 12']),
            [
                Section('2.1.1 Fine details', ['Detailed information', 'Even more details']),
                Section('2.1.2 Extra details', ['Fine as frog hair', 'Fine as espresso grounds'])
            ],
            Section('2.2 Subsection B', ['Paragraph 13 Paragraph 14'])
        ],
        Section('3. Section Three', ['Paragraph 15 Paragraph 16'])
    ]