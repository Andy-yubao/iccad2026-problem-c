import sys
print('E28 stdout before stderr', flush=True)
print('E28 harmless stderr', file=sys.stderr, flush=True)
print('E28 stdout after stderr', flush=True)
