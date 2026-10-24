import sys

from coact_cps.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "demo")
    main()
