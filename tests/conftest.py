import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ANIATE_BANNER", "0")

import aniate  # noqa: E402

aniate.verbosity("quiet")
