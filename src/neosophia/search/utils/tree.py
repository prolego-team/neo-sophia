from typing import Callable, Any


OrderedTree = list[Any | list]


EOL_TOKEN = '<END OF LIST>'


def parse_recursive(input_list: list[tuple[int,Any]], depth: int) -> OrderedTree:
    """Takes a list of strings and parses it into an ordered tree.

    There are assumptions:
    1. Levels can only increase by one.
    2. Levels can decrease by an arbitrary amount.
    """

    output_tree = []
    while len(input_list)>1:
        head_depth, text = input_list[0]

        # add the first element of the list
        if head_depth<depth:
            return output_tree, input_list
        output_tree.append(text)

        # look ahead
        if input_list[1]==EOL_TOKEN:
            return output_tree, []

        next_depth = input_list[1][0]
        if next_depth==depth:
            input_list = input_list[1:]
        elif next_depth>depth:
            child_tree, input_list = parse_recursive(input_list[1:], next_depth)
            output_tree.append(child_tree)
        elif next_depth<depth:
            return output_tree, input_list[1:]

    return output_tree, input_list


def parse(lst_in: list[tuple[int,Any]]) -> OrderedTree:
    """Turn a flat list with levels into an ordered tree."""
    lst_in = lst_in + [(-1,EOL_TOKEN)]
    output, _ = parse_recursive(lst_in, 1)
    return output


# Move index relative to a position
def move_right(ind):
    right_ind = list(ind)
    right_ind[-1] += 1
    return tuple(right_ind)

def move_left(ind):
    left_ind = list(ind)
    left_ind[-1] -= 1
    return tuple(left_ind)

def move_up(ind):
    """Up a level and left"""
    if len(ind)>1:
        return move_left(ind[:-1])
    else:
        return None

def move_down(ind):
    """Right and down a level"""
    subsection_ind = move_right(ind) + (0,)
    return tuple(subsection_ind)



def flatten(tree: OrderedTree, current_level: int = 1) -> tuple:
    """Flatten a tree and return the levels and nodes."""
    levels = []
    nodes = []
    for node in tree:
        match node:
            case list():
                sub_levels, sub_nodes = flatten(node, current_level+1)
                levels += sub_levels
                nodes += sub_nodes
            case _:
                levels.append(current_level)
                nodes.append(node)

    return levels, nodes



def search(tree_in: OrderedTree, cond: Callable):
    """Recursively search an ordered tree and return the node that satisfy
    cond(node)."""
    for node in tree_in:
        match node:
            case list():
                yield from search(node, cond)
            case _:
                if cond(node):
                    yield node


def search_ind(tree_in: OrderedTree, cond: Callable, ind_prefix=None):
    """Recursively search an ordered tree and return the indices that satisfy
    cond(node)."""
    ind_prefix = [] if ind_prefix is None else ind_prefix
    for i,node in enumerate(tree_in):
        match node:
            case list():
                yield from search_ind(node, cond, ind_prefix + [i])
            case _:
                if cond(node):
                    yield tuple(ind_prefix + [i])


def get_from_tree(tree: OrderedTree, indices: tuple[int, ...]):
    """Get a node from the tree with indices (i1, i2, ...)."""
    ind = indices[0]
    if (ind<0) or (ind>=len(tree)) or (len(indices)==0):
        return None

    if len(indices)==1:
        return tree[ind]
    else:
        if not isinstance(tree[ind], list):
            return None

        return get_from_tree(tree[ind], indices[1:])


def grow(tree: OrderedTree, grow_func: Callable):
    """grow_func : node -> list[Any]"""
    new_tree = []
    for node in tree:
        match node:
            case list():
                new_node = grow(node, grow_func)
            case _:
                new_node = grow_func(node)

        new_tree.append(new_node)

    return new_tree


def transform(tree: OrderedTree, transformation: Callable) -> OrderedTree:
    """Transform a tree by calling transformation on every node."""
    new_tree = []
    for node in tree:
        match node:
            case list():
                new_node = transform(node, transformation)
            case _:
                new_node = transformation(node)
        new_tree.append(new_node)

    return new_tree