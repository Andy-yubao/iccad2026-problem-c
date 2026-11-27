"""Independent copy of the current exact four-tier scheduler."""
LR_LEVELS=(1e-3,2.5e-4,1e-4,5e-5)

class PlateauController:
    def __init__(self,optimizer):
        self.optimizer=optimizer;self.best=None;self.lr_bad=0;self.early_bad=0
        self.drops=0;self.last_drop_epoch=None;self.lowest_completed_epochs=0
    def step(self,value,epoch):
        if epoch<15:return
        if self.drops==3 and epoch>self.last_drop_epoch:self.lowest_completed_epochs=epoch-self.last_drop_epoch
        delta=max(1e-5,abs(self.best)*1e-3) if self.best is not None else 0.
        if self.best is None or value<self.best-delta:self.best=value;self.lr_bad=0;self.early_bad=0
        else:self.lr_bad+=1;self.early_bad+=1
        if self.lr_bad>=3 and self.drops<3:
            self.drops+=1
            for group in self.optimizer.param_groups:group['lr']=LR_LEVELS[self.drops]
            self.lr_bad=0;self.last_drop_epoch=epoch;self.lowest_completed_epochs=0
    def should_stop(self):return self.early_bad>=3 and self.drops==3 and self.lowest_completed_epochs>=2
    def state_dict(self):return {k:v for k,v in vars(self).items() if k!='optimizer'}
    def load_state_dict(self,state):
        if set(state)!=set(self.state_dict()):raise ValueError('Scheduler identity mismatch')
        for k,v in state.items():setattr(self,k,v)
