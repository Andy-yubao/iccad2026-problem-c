"""Prepared single-arm entry. Disabled until the user authorizes training."""
import argparse
from .control import ARMS, assert_training_authorized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert_training_authorized(args.arm)
    from .runtime import run
    run(args)


if __name__ == '__main__':
    main()
