import torch

class DataBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []

    def add(self, state, action, next_state):
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.float32))
        self.next_states.append(torch.tensor(next_state, dtype=torch.float32))

    def get(self):
        X = torch.stack(self.states)
        U = torch.stack(self.actions)
        Y = torch.stack(self.next_states)
        if U.ndim == 1:
            U = U.unsqueeze(1)
        return X, U, Y
