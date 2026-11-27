"""E169 plateau rules with a separate minimum LR for nonfinal stages."""
from .loss_config import PLAN

LR = tuple(PLAN["lr_levels"])

class PlateauController:
    def __init__(self, optimizer, stage):
        self.optimizer = optimizer
        self.final = stage == len(PLAN["stages"])
        self.last_level = len(LR) - 1 if self.final else LR.index(PLAN["nonfinal_min_lr"])
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
        if self.lr_bad >= 3 and self.drops < self.last_level:
            self.drops += 1
            self.last_drop_epoch = epoch
            self.lr_bad = self.lowest_completed_epochs = 0
            for group in self.optimizer.param_groups:
                group['lr'] = LR[self.drops] * group.get('lr_scale', 1.)

    def should_stop(self):
        if not self.final:
            return self.drops == self.last_level and self.lr_bad >= 3
        return self.drops == self.last_level and self.early_bad >= 3 and self.lowest_completed_epochs >= 2

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('Scheduler schema mismatch')
        self.__dict__.update(state)
