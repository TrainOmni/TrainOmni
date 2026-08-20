"""Allow ``python -m trainomni`` to invoke the control-plane CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
