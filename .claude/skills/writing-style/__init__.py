"""Score prose against published scientific writing. Stdlib only.

    from writingstyle import report
    print(report(open("draft.md").read()))
"""
from .check import measure, report  # noqa: F401

__version__ = "0.1.0"
