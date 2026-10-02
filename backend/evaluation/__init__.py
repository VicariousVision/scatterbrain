"""RAGAS-based evaluation of the Scatterbrain RAG pipeline.

Importing this package disables RAGAS usage analytics before RAGAS itself is
imported. RAGAS otherwise posts anonymous usage events to a remote endpoint,
which conflicts with Scatterbrain's fully local, no-external-calls design.
"""

import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
