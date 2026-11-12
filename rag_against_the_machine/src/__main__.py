""" __main__ entry file """

# pylint: disable=broad-exception-caught

import traceback
import fire

from src.cli.cli import CLI


if __name__ == '__main__':

    try:
        fire.Fire(CLI)
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Chunking failed: {type(e).__name__}: {e}")
        traceback.print_exc()
