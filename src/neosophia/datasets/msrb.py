"""
Dataclasses for MSRB rules.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.

from typing import Any, List
from dataclasses import dataclass

from neosophia.llmtools.util import Colors, colorize


@dataclass
class Rule:
    """MSRB rule, including various text"""

    uid: str
    description: str
    sections: List[Any]
    interpretations: List[Any]
    amendments: List[Any]

    def __str__(self):
        """format a string representation of the rule"""

        a = colorize('uid: ', Colors.GREEN) + self.uid + '\n'
        b = colorize('description: ', Colors.GREEN) + self.description + '\n'
        c = colorize('sections: ', Colors.GREEN)
        for s in self.sections:
            c += str(s) + '\n'
        d = colorize('interpretations: ', Colors.GREEN)
        for s in self.interpretations:
            d += s + '\n'
        e = colorize('amendments: ', Colors.GREEN)
        for s in self.amendments:
            e += s + '\n'

        return a + b + c + d + e
