import argparse
import logging

from . import candumpgen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic candump log files.")

    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0,
                        help="Increase output verbosity, up to three times.")

    parser.add_argument("dbc_file", metavar="DBC_FILE", type=str,
                        help="The DBC file to generate a realistic candump off of.")

    parser.add_argument("interface", metavar="INTERFACE", type=str,
                        help="The interface to simulate/to open a socket to, e.g. vcan0.")

    parser.add_argument("-f", "--candump-file", dest="candump_file", type=str, default=None,
                        help="The name/path of the candump file to generate.")

    args = parser.parse_args()

    log_level = logging.ERROR
    if args.verbose > 0:
        log_level = logging.WARNING
    if args.verbose > 1:
        log_level = logging.INFO
    if args.verbose > 2:
        log_level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=log_level)

    if args.candump_file is None:
        candumpgen.cangen(args.dbc_file, args.interface)
    else:
        candumpgen.generate_candump(args.candump_file, args.dbc_file, args.interface)


if __name__ == "__main__":
    main()
