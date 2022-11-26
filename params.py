class Parameters():
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.replay_length = 1000
        self.eps_max = 1.0
        self.eps_min = 0.05
        self.eps_decay = (self.eps_max - self.eps_min)/(self.replay_length)
        self.update_frequency = 4
        self.pre_train_steps = 500
        self.num_episodes = 100000
        self.gamma = 0.99