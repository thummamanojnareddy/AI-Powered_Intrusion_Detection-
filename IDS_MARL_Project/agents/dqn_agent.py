import random
import numpy as np

class DQNAgent:
    def __init__(self, action_size=4):
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

    def act(self, state):
        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation based on anomaly score
        anomaly_score = state[0]

        if anomaly_score > 0.65:
            return 2   # Block
        elif anomaly_score > 0.45:
            return 1   # Alert
        else:
            return 0   # Allow

    def learn(self):
        # Reduce randomness gradually
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay