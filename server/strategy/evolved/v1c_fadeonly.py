"""v1c_fadeonly: same detector as v1a, but evaluated with enable_flip=False.

The detector is literally v1a. The difference is in the harness config,
enabled via a wrapper param. Kept as a separate variant so the comparison
table shows both with/without flip side-by-side.

(The actual harness flag is set by run_comparison; this module just
re-exports v1a's detector so both point to the same detection logic.)
"""

from .v1a_filtered import detect_lines  # noqa: F401
