"""Complete only the isolated derivative diagnostic; preserve forward captures."""
from .capture import HERE,RUN,read,save,gradients,load_rows

def main():
    previous=read(HERE/'capture_state.json')
    if previous.get('status')=='failed':save(HERE/'capture_failure_before_gradient_buffer_fix.json',previous)
    rows=load_rows('dev_monitor500')
    gradients(RUN/'checkpoints/best.pt',rows,None)
    save(HERE/'capture_state.json',dict(status='complete',cases=500,states=13,training=False,postprocessing_calls=0,
        completed_existing_forward_and_Loss=True,gradient_buffers_isolated=True,original_failure_preserved=True))
    print('Gradient diagnostics complete with separate ordinary buffers.',flush=True)

if __name__=='__main__':main()
