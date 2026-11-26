""" __main__ entry file """

# pylint: disable=broad-exception-caught

import sys
import traceback

import fire

from src.cli.cli import CLI


def main() -> None:
    """Run the Fire CLI and translate application failures."""
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("Operation cancelled.", file=sys.stderr)
        raise SystemExit(130) from None
    except (FileNotFoundError, NotADirectoryError) as error:
        print(
            f"Required input path unavailable: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except OSError as error:
        print(f"Input/output error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        print(
            f"Operation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
