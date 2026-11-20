"""Single scratch FM experiment; launch interlock checked before data/CUDA."""
import argparse
from .control import assert_training_authorized
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['fm'],default='fm');p.add_argument('--resume',action='store_true')
    args=p.parse_args();assert_training_authorized(args.arm)
    from .runtime import run
    run(args)
if __name__=='__main__':main()
