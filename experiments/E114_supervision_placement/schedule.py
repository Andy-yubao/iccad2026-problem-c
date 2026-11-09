"""Scratch schedule: 14 protected epochs; same fixed objective everywhere."""
LR = (1e-3, 2.5e-4, 1e-4, 5e-5)

class PlateauController:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.best = None
        self.lr_bad = self.early_bad = self.drops = 0
        self.last_drop_epoch = None
        self.lowest_completed_epochs = 0

    def step(self, value, epoch):
        if epoch < 15:
            return
        if self.drops == len(LR)-1:
            self.lowest_completed_epochs = epoch-self.last_drop_epoch
        delta = max(1e-5, abs(self.best)*.001) if self.best is not None else 0.
        if self.best is None or value < self.best-delta:
            self.best = value
            self.lr_bad = self.early_bad = 0
        else:
            self.lr_bad += 1
            self.early_bad += 1
        if self.lr_bad >= 3 and self.drops < len(LR)-1:
            self.drops += 1
            self.last_drop_epoch = epoch
            self.lr_bad = self.lowest_completed_epochs = 0
            for group in self.optimizer.param_groups:
                group['lr'] = LR[self.drops]

    def should_stop(self):
        return self.drops == len(LR)-1 and self.early_bad >= 3 and self.lowest_completed_epochs >= 2

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('Scheduler schema mismatch')
        self.__dict__.update(state)
