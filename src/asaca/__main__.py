
import sys
from asaca.cli import main as cli_main

if __name__ == "__main__":
    # forward everything after "-m asaca" to the real CLI
    sys.exit(cli_main(sys.argv[1:]))
