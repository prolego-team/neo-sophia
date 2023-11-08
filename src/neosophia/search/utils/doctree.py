"""Store a document as a tree."""

from typing import Any, Callable
from dataclasses import dataclass
from copy import copy
import os
import yaml

from neosophia.search.utils import tree
from neosophia.text_utils import words_in_list, combine_strings


@dataclass
class Section:
    """A document section."""
    title: str
    contents: list[Any]
    metadata: dict | None = None

    def __repr__(self) -> str:
        return (
            f'{self.title}: '
            f'{len(self.contents)} paragraphs, '
            f'{words_in_list(self.contents)} words'
        )


DocTree = list[Section | list]
Index = tree.Index


def parse(sections: list[Section], levels: list[int]) -> DocTree:
    """Parse levels and sections into a DocTree."""
    return tree.parse(list(zip(levels, sections)))


def search(doc_tree: DocTree, search_func: Callable) -> list[Index]:
    """Search a DocTree, returning indices of hits."""
    return list(tree.search_ind(doc_tree, search_func))


def flatten_doctree(tree_inp: DocTree, ind_prefix=None) -> tuple[Index, Any]:
    """DocTree -> list of chunks with tree indices.

    Note that an extra int is added to the indices indicating the element
    within that sections contents list."""
    ind_prefix = [] if ind_prefix is None else ind_prefix
    for i,node in enumerate(tree_inp):
        match node:
            case list():
                yield from flatten_doctree(node, ind_prefix + [i])
            case Section():
                for j,item in enumerate(node.contents):
                    yield tuple(ind_prefix + [i,j]), item


def get_supersection(doc_tree: DocTree, ind: tuple[int]) -> str | None:
    """Get the supersection of an index, i.e. the last string in the
    contents of a section directly above the given index."""
    super_ind = tree.move_up(ind)
    super_section = None
    if super_ind is not None:
        super_section = tree.get_from_tree(doc_tree, super_ind)
        super_section = (
            super_section.contents[-1]
            if (isinstance(super_section, Section) and len(super_section.contents)>0)
            else None
        )

    return super_section


def get_subsection(doc_tree: DocTree, ind: tuple[int]) -> str | None:
    """Get the subsection of an index, i.e. the first string in the
    contents of a section directly below the given index."""
    sub_ind = tree.move_down(ind)
    sub_section = None
    if sub_ind is not None:
        sub_section = tree.get_from_tree(doc_tree, sub_ind)
        sub_section = (
            sub_section.contents[0]
            if (isinstance(sub_section, Section) and len(sub_section.contents)>0)
            else None
        )

    return sub_section


def expand(item: str, doc_tree: DocTree, index: tuple[int]) -> list[str]:
    """Expand on an item to include the super and sub sections of an index."""
    super_section = get_supersection(doc_tree, index)
    sub_section = get_subsection(doc_tree, index)

    has_super_section = super_section is not None
    has_sub_section = sub_section is not None
    expanded_items = [
        item,
        super_section+' '+item if has_super_section else None,
        item+' '+sub_section if has_sub_section else None,
        super_section+' '+item+' '+sub_section if has_super_section and has_sub_section else None
    ]
    expanded_items = [item for item in expanded_items if item is not None]

    return expanded_items


def show_tree(tree_inp: DocTree, indent: str = '') -> str:
    """Return a string representation of a DocTree."""
    output = ''
    for node in tree_inp:
        match node:
            case list():
                output += show_tree(node, indent+'  ')
            case _:
                output += indent + str(node) + '\n'

    return output


def get_tree_text(
        tree_inp: DocTree,
        start_level: int = 0,
        end_level: int | None = None,
        current_level: int = 0
    ) -> str:
    """Get all of the text from a DocTree within certain levels."""
    output = ''
    end_level = end_level if end_level is not None else 1000000
    if current_level>end_level:
        return ''
    if current_level<start_level:
        for node in tree_inp:
            match node:
                case list():
                    output += get_tree_text(node, start_level, end_level, current_level+1)
                case Section():
                    pass
    else:
        for node in tree_inp:
            match node:
                case list():
                    output += get_tree_text(node, start_level, end_level, current_level+1)
                case Section():
                    contents = '\n'.join(node.contents)
                    output += f'{node.title}\n{contents}\n'


    return output



def consolidate_leaves(tree_inp: DocTree, word_thresh: int = 200) -> DocTree:
    """Combine multiple small leaf sections."""

    new_tree = []
    for node in tree_inp:
        match node:
            case list():
                # If this node consists of a homogenous list of Sections then we may be
                # able to consolidate
                if all(isinstance(subnode, Section) for subnode in node):
                    word_length = sum(
                        words_in_list(section.contents) for section in node
                    )
                    if word_length<=word_thresh:
                        # Consolidate
                        new_node = [Section(
                            title=f'{node[0].title}-{node[-1].title}',
                            contents=[text for section in node for text in section.contents],
                            metadata=node[0].metadata
                        ),]
                    else:
                        # Can't consolidate
                        new_node = consolidate_leaves(node, word_thresh)
                else:
                    # Non-homogenous list
                    new_node = consolidate_leaves(node, word_thresh)

            case Section():
                new_node = copy(node)
        new_tree.append(new_node)

    return new_tree


def consolidate_paragraphs(tree_inp: DocTree, word_limit: int = 100, sep=' ') -> DocTree:
    """Combine Section paragraphs to avoid short paragraphs."""
    new_tree = []
    for node in tree_inp:
        match node:
            case list():
                new_node = consolidate_paragraphs(node, word_limit)
            case Section():
                new_node = Section(
                    node.title,
                    combine_strings(node.contents, word_limit, sep),
                    node.metadata
                )
        new_tree.append(new_node)

    return new_tree




def section_repr(dumper, data):
    """YAML represection of a Section."""
    return dumper.represent_mapping('!section', vars(data))


def section_constructor(loader, node):
    """Construct a Section from its YAML representation."""
    values = loader.construct_mapping(node)
    return Section(**values)


yaml.add_representer(Section, section_repr)
yaml.SafeLoader.add_constructor('!section', section_constructor)

def write(filename: str, tree_inp: DocTree) -> None:
    """Write doc tree to file."""
    with open(filename, 'w') as f:
        yaml.dump(tree_inp, f)


def read(filename: str) -> DocTree:
    """Read a doc tree from file."""
    if not os.path.exists(filename):
        assert FileNotFoundError(f'Can\'t find file {filename}')

    with open(filename, 'r') as f:
        tree_inp = yaml.safe_load(f)

    return tree_inp
