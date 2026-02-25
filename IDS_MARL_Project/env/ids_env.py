import gym
import numpy as np
import random

class IDSEnv(gym.Env):
    def __init__(self):
        super(IDSEnv, self).__init__()

        # Action space
        # 0 = Allow
        # 1 = Alert
        # 2 = Block
        # 3 = Isolate
        self.action_space = gym.spaces.Discrete(4)

        # State: 4 network features
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Multi-cloud providers
        self.cloud_providers = ["AWS", "Azure", "GCP"]

        self.state = None
        self.current_cloud = None

    def reset(self):
        # Generate random network state
        self.state = np.random.rand(4).astype(np.float32)

        # Random cloud source
        self.current_cloud = random.choice(self.cloud_providers)

        return self.state

    def step(self, action):
        anomaly_score = self.state[0]

        # Ground truth attack condition
        attack = anomaly_score > 0.6

        # Reward logic
        if attack and action in [1, 2, 3]:
            reward = 10
        elif attack and action == 0:
            reward = -10
        elif not attack and action == 0:
            reward = 5
        else:
            reward = -5

        done = False

        # Next state
        self.state = np.random.rand(4).astype(np.float32)

        # Next cloud traffic
        self.current_cloud = random.choice(self.cloud_providers)

        # Return cloud info in info dict
        info = {
            "cloud": self.current_cloud,
            "attack": attack
        }

        return self.state, reward, done, info