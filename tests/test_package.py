"""The smallest test that proves the toolchain is wired up end to end.

It also guards the version wiring: hatchling reads ``__version__`` out of
``src/neuronest/__init__.py`` to build the wheel, so a malformed value there
breaks packaging rather than anything you would notice at runtime.
"""

import neuronest


def test_version_is_three_integers() -> None:
    parts = neuronest.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
