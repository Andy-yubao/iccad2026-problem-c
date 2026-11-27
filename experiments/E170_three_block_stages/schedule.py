"""Fixed dev500 plateau rules for frozen-stage and joint training."""
from .loss_config import PLAN


def learning_rates(joint):
    return tuple(PLAN['joint']['lr_levels'] if joint else PLAN['lr_levels'])


class PlateauController:
    def __init__(self, optimizer, joint):
        self.optimizer = optimizer
        self.rates = learning_rates(joint)
        self.first_epoch = PLAN['joint']['baseline_epoch'] if joint else PLAN['plateau']['first_stage_epoch']
        self.best = None
        self.bad_epochs = 0
        self.level = 0

    def step(self, value, epoch):
        if epoch < self.first_epoch:
            return
        plateau = PLAN['plateau']
        delta = (max(plateau['min_absolute_improvement'], abs(self.best) * plateau['min_relative_improvement'])
                 if self.best is not None else 0.)
        if self.best is None or value < self.best - delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= plateau['patience_epochs'] and self.level < len(self.rates) - 1:
            self.level += 1
            self.bad_epochs = 0
            for group in self.optimizer.param_groups:
                group['lr'] = self.rates[self.level] * group.get('lr_scale', 1.)

    def should_stop(self):
        return self.level == len(self.rates) - 1 and self.bad_epochs >= PLAN['plateau']['patience_epochs']

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('Scheduler schema mismatch')
        self.__dict__.update(state)
