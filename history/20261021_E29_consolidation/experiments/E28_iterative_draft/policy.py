"""Four user-specified LR levels, with two complete lowest-tier epochs."""
LR_LEVELS = (1e-3, 2.5e-4, 1e-4, 5e-5)


class PlateauController:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.best = None
        self.lr_bad = 0
        self.early_bad = 0
        self.drops = 0
        self.last_drop_epoch = None
        self.lowest_epochs = 0

    def step(self, value, epoch):
        if self.drops == 3 and epoch > self.last_drop_epoch:
            self.lowest_epochs = epoch - self.last_drop_epoch
        if epoch < 15:
            return
        delta = max(1e-5, abs(self.best) * .001) if self.best is not None else 0.
        if self.best is None or value < self.best - delta:
            self.best, self.lr_bad, self.early_bad = value, 0, 0
        else:
            self.lr_bad += 1; self.early_bad += 1
        if self.lr_bad >= 3 and self.drops < 3:
            self.drops += 1
            for group in self.optimizer.param_groups:
                group['lr'] = LR_LEVELS[self.drops]
            self.lr_bad = 0; self.last_drop_epoch = epoch; self.lowest_epochs = 0

    def should_stop(self):
        return self.early_bad >= 3 and self.drops == 3 and self.lowest_epochs >= 2

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('Scheduler state mismatch')
        for key, value in state.items():
            setattr(self, key, value)
