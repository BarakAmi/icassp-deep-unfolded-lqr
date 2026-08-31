"""`python -m mbl.cli`, equivalent to the installed `mbl` entry point."""

from __future__ import annotations

from .app import main

raise SystemExit(main())
