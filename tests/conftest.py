"""Put LLM_LTB's ``LLM_AMS`` directory and this repository on ``sys.path``.

LLM_LTB is not an installable package; its import root is ``LLM_AMS/``. Point to it with
``$LTB_LLM_AMS`` or keep the default layout (``../LLM_LTB/LLM_AMS`` next to this repo).
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LTB = os.path.abspath(os.environ.get("LTB_LLM_AMS")
                      or os.path.join(os.path.dirname(REPO), "LLM_LTB", "LLM_AMS"))
if not os.path.isfile(os.path.join(LTB, "agent", "extensions.py")):
    raise RuntimeError(f"LLM_LTB with route extensions not found at {LTB}; set LTB_LLM_AMS")
for p in (REPO, LTB):
    if p not in sys.path:
        sys.path.insert(0, p)
