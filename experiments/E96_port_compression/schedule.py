"""Warm independent encoder: epoch0 baseline, no protection; fixed four tiers."""
LEVELS=(1e-3,2.5e-4,1e-4,5e-5)

class Scheduler:
    def __init__(self):
        self.best=None;self.bad=0;self.early_bad=0;self.drops=0;self.last_drop=None
    def step(self,value,epoch,opt):
        delta=max(1e-5,abs(self.best)*.001) if self.best is not None else 0.
        if self.best is None or value<self.best-delta:
            self.best=value;self.bad=0;self.early_bad=0
        else:self.bad+=1;self.early_bad+=1
        if self.bad>=3 and self.drops<3:
            self.drops+=1;self.bad=0;self.last_drop=epoch
            for group in opt.param_groups:group['lr']=LEVELS[self.drops]
    def stop(self,epoch):return self.drops==3 and self.early_bad>=3 and epoch-self.last_drop>=2
