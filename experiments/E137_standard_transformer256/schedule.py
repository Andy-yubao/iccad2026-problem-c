"""One-epoch accepted-step warmup, protected epochs, fixed final-Loss plateau."""
LR = (3e-4, 1e-4, 3e-5, 1e-5)


def improved(value, best):
    return best is None or value < best - max(1e-5, abs(best) * 1e-4)


class PlateauController:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.best = None
        self.lr_bad = self.early_bad = self.drops = 0
        self.last_drop_epoch = None
        self.lowest_completed_epochs = 0

    def before_update(self, accepted_steps, steps_per_epoch):
        rate = (3e-5 + (3e-4 - 3e-5) * accepted_steps / max(steps_per_epoch - 1, 1)
                if accepted_steps < steps_per_epoch else LR[self.drops])
        for group in self.optimizer.param_groups:
            group['lr'] = rate

    def step(self, value, epoch):
        if improved(value, self.best):
            self.best = value
            self.lr_bad = self.early_bad = 0
        elif epoch >= 15:
            self.lr_bad += 1
            self.early_bad += 1
        if epoch < 15:
            return
        if self.drops == len(LR) - 1:
            self.lowest_completed_epochs = epoch - self.last_drop_epoch
        if self.lr_bad >= 6 and self.drops < len(LR) - 1:
            self.drops += 1
            self.last_drop_epoch = epoch
            self.lr_bad = self.early_bad = self.lowest_completed_epochs = 0
            for group in self.optimizer.param_groups:
                group['lr'] = LR[self.drops]

    def should_stop(self):
        return self.drops == len(LR) - 1 and self.early_bad >= 8 and self.lowest_completed_epochs >= 8

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('Scheduler schema mismatch')
        self.__dict__.update(state)
