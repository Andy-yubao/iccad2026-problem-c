"""Request the same safe boundary pause as Ctrl+C, without a foreground console."""
import argparse
import datetime
import json
from .io import HERE,save

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pause',action='store_true',required=True)
    parser.parse_args()
    state=json.loads((HERE/'queue_state.json').read_text(encoding='utf-8'))
    if state['status']!='running':raise RuntimeError('Queue is not running: '+state['status'])
    save(HERE/'PAUSE_REQUEST',dict(queue_id=state['queue_id'],
        requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Explicit user safe pause request'))
    print('Pause requested. Wait for queue_state.json status=paused before --resume.')

if __name__=='__main__':main()
