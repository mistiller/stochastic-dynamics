from dataclasses import dataclass
import numpy as np

@dataclass
class Dataset:
    t:np.array
    input:np.array
    cost:np.array
    benefit:np.array

    def validate(self):
        if not isinstance(self.t, np.ndarray):
            raise TypeError("t must be an a numpy array")
        if not isinstance(self.input, np.ndarray):
            raise TypeError("input must be a numpy array")
        if not isinstance(self.cost, np.ndarray):
            raise TypeError("cost must be a numpy array")
        if not isinstance(self.benefit, np.ndarray):
            raise TypeError("benefit must be a numpy array")

        if len(self.t) != len(self.input):
            raise ValueError(f"t and input must have the same length ({len(self.t)}:{len(self.input)})")
        if len(self.t) != len(self.cost):
            raise ValueError("t and cost must have the same length")
        if len(self.t) != len(self.benefit):
            raise ValueError("t and benefit must have the same length")

    def __post_init__(self):
        self.validate()

    def arrays(self):
        return (self.t, self.input, self.cost, self.benefit)