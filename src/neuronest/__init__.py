"""NeuroNest — a RAG service that answers only from retrieved context.

The package is deliberately thin at the top level. Import the pieces you need
from their own modules rather than re-exporting them here, so that importing
``neuronest`` never drags in a model or a database client as a side effect.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
