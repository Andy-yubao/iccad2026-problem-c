"""Serialisable phase-local plateau controller. Second P event requests F."""
LR = {'P': (1e-3,2.5e-4), 'F': (2.5e-4,1e-4,5e-5)}

class PlateauController:
    def __init__(self, optimizer, phase='P'):
        self.optimizer=optimizer
        self.phase=phase
        self.best=None
        self.lr_bad=0
        self.early_bad=0
        self.drops=0
        self.last_drop_epoch=None
        self.lowest_completed_epochs=0
        self.transition_pending=False

    def step(self, value, epoch):
        if self.phase=='P' and epoch<15:
            return
        if self.phase=='F' and self.drops==2:
            self.lowest_completed_epochs=epoch-self.last_drop_epoch
        delta=max(1e-5,abs(self.best)*.001) if self.best is not None else 0.
        if self.best is None or value<self.best-delta:
            self.best=value
            self.lr_bad=self.early_bad=0
        else:
            self.lr_bad+=1
            self.early_bad+=1
        if self.lr_bad>=3:
            if self.phase=='P' and self.drops==1:
                self.transition_pending=True
                return
            if self.drops<len(LR[self.phase])-1:
                self.drops+=1
                for group in self.optimizer.param_groups:
                    group['lr']=LR[self.phase][self.drops]
                self.lr_bad=0
                self.last_drop_epoch=epoch
                self.lowest_completed_epochs=0

    def should_stop(self):
        return self.phase=='F' and self.drops==2 and self.early_bad>=3 and self.lowest_completed_epochs>=2

    def state_dict(self):
        return {k:v for k,v in self.__dict__.items() if k!='optimizer'}

    def load_state_dict(self, state):
        if set(state)!=set(self.state_dict()) or state['phase']!=self.phase:
            raise ValueError('Phase/scheduler state mismatch')
        self.__dict__.update(state)
