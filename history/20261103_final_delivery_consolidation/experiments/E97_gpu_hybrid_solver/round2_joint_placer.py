"""Registered L-last-four/F joint stage; no architecture or input changes."""
from .student import DeploymentModel as StudentDeployment

class DeploymentModel(StudentDeployment):
    def configure_trainable(self):
        self.requires_grad_(False);self.refiner.requires_grad_(True)
        for layer in self.placer.layers[-4:]:layer.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode)
        for layer in self.placer.layers[-4:]:layer.train(mode)
        return self
    def forward(self,g):
        return self.refiner(self.placer(g),g)
