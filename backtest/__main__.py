"""Allow ``python -m backtest`` to invoke the CLI."""

from backtest.cli import main

raise SystemExit(main())
