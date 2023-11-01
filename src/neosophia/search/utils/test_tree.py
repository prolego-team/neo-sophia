from neosophia.search.utils.tree import (
    parse,
    move_up, move_down, move_left, move_right,
    search, search_ind, get_from_tree, transform, flatten
)

def ordered_list():
    flat_input = [
        (1, '1'),
        (1, '2'),
        (2, '2.1'),
        (2, '2.2'),
        (2, '2.3'),
        (1, '3'),
        (1, '4'),
        (2, '4.1'),
        (3, '4.1.1'),
        (3, '4.1.2'),
        (1, '5'),
    ]
    ordered_tree = [
        '1',
        '2',
        [
            '2.1',
            '2.2',
            '2.3'
        ],
        '3',
        '4',
        [
            '4.1',
            [
                '4.1.1',
                '4.1.2'
            ]
        ],
        '5'
    ]
    return flat_input, ordered_tree


def test_parse():
    flat_input, target = ordered_list()
    assert parse(flat_input)==target


def test_move():
    _, tree = ordered_list()
    ind = (0,)
    assert get_from_tree(tree, ind)=='1'
    assert get_from_tree(tree, move_up(ind))==None
    assert get_from_tree(tree, move_right(ind))=='2'
    assert get_from_tree(tree, move_left(ind))==None
    assert get_from_tree(tree, move_down(ind))==None

    ind = (2,1)
    assert get_from_tree(tree, ind)=='2.2'
    assert get_from_tree(tree, move_up(ind))=='2'
    assert get_from_tree(tree, move_right(ind))=='2.3'
    assert get_from_tree(tree, move_left(ind))=='2.1'
    assert get_from_tree(tree, move_down(ind))==None

    ind = (5,0)
    assert get_from_tree(tree, ind)=='4.1'
    assert get_from_tree(tree, move_up(ind))=='4'
    assert get_from_tree(tree, move_right(ind))==['4.1.1', '4.1.2']
    assert get_from_tree(tree, move_left(ind))==None
    assert get_from_tree(tree, move_down(ind))=='4.1.1'

    ind = (5,1,1)
    assert get_from_tree(tree, ind)=='4.1.2'
    assert get_from_tree(tree, move_up(ind))=='4.1'
    assert get_from_tree(tree, move_right(ind))==None
    assert get_from_tree(tree, move_left(ind))=='4.1.1'
    assert get_from_tree(tree, move_down(ind))==None


def test_flatten():
    target, ordered_tree = ordered_list()
    target_levels, target_nodes = zip(*target)

    levels, nodes = flatten(ordered_tree)
    assert nodes==list(target_nodes)
    assert levels==list(target_levels)

    assert parse(list(zip(levels,nodes)))==ordered_tree


def test_search():
    _, ordered_tree = ordered_list()

    search_f = lambda inp: inp=='4.1.2'
    results = [res for res in search(ordered_tree, search_f)]
    assert results[0]=='4.1.2'
    assert len(results)==1

    search_f = lambda inp: inp=='4.2.1'
    results = [res for res in search(ordered_tree, search_f)]
    assert len(results)==0


def test_search_ind():
    _, ordered_tree = ordered_list()

    search_f = lambda inp: inp=='4.1.2'
    results = [res for res in search_ind(ordered_tree, search_f)]
    assert results[0]==(5,1,1)
    assert len(results)==1

    search_f = lambda inp: inp=='4.2.1'
    results = [res for res in search_ind(ordered_tree, search_f)]
    assert len(results)==0


def test_get_from_tree():
    _, ordered_tree = ordered_list()

    assert get_from_tree(ordered_tree, (5,1,1))=='4.1.2'
    assert get_from_tree(ordered_tree, (0,))=='1'
    assert get_from_tree(ordered_tree, (2,1))=='2.2'


def test_transform():
    _, ordered_tree = ordered_list()

    target = [
        '1',
        '2',
        [
            '1',
            '2',
            '3'
        ],
        '3',
        '4',
        [
            '1',
            [
                '1',
                '2'
            ]
        ],
        '5'
    ]
    xform = lambda inp: inp.split('.')[-1]
    assert transform(ordered_tree, xform)==target