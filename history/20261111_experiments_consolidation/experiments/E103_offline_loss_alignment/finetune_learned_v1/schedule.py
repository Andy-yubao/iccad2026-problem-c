"""Epoch0 plateau baseline, zero protection, normal early stopping without caps."""
TIERS=(1e-3,2.5e-4,1e-4,5e-5)

class PhaseSchedule:
    def __init__(self):
        self.lr=TIERS[0];self.tier=0;self.best=None;self.bad=0;self.last_drop_epoch=None

    def step(self,value,epoch):
        if self.best is None:
            self.best=value;return False
        delta=max(1e-5,abs(self.best)*.001)
        if value<self.best-delta:self.best=value;self.bad=0
        else:self.bad+=1
        if self.bad<3:return False
        if self.tier<len(TIERS)-1:
            self.tier+=1;self.lr=TIERS[self.tier]
            self.last_drop_epoch=epoch;self.bad=0;self.best=value
            return False
        return True

    def state_dict(self):return dict(self.__dict__)
    def load_state_dict(self,state):
        assert set(state)==set(self.__dict__)
        self.__dict__.update(state)
