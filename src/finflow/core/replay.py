import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.prev_acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.size = size; self.ptr = 0; self.full = False

    def store(self, o, a, r, o2, d, prev_a):
        self.obs[self.ptr] = o
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d
        self.prev_acts[self.ptr] = prev_a
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0: self.full = True

    def sample_batch(self, batch_size):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch_size)
        return dict(
            obs=self.obs[idx],
            acts=self.acts[idx],
            rews=self.rews[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            prev_acts=self.prev_acts[idx],
        )

    def __len__(self):
        return self.size if self.full else self.ptr
