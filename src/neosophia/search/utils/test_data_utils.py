from neosophia.search.utils import data_utils as du


def test_reciprocal_rank_fusion():
    test_input = [
        ['A','B','C'],
        ['C','A','B'],
        ['A','C','B'],
        ['A','C','B'],
    ]

    assert du.reciprocal_rank_fusion(test_input)==['A','C','B']