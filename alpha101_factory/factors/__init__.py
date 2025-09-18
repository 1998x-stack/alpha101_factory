# -*- coding: utf-8 -*-
"""
Import factor modules so their @register decorators run.
If you add new files inside this package, either import them here
or rely on registry.ensure_loaded() which will auto-discover too.
"""
# Explicit imports (safe & fast)
from . import alphas_basic  # noqa: F401
try:
    from . import alphas_more  # noqa: F401
except Exception:
    pass
