import gym
import numpy as np

class IDSEnv(gym.Env):
    def __init__(self):
        super(IDSEnv, self).__init__()

        # Actions:
        # 0 = Allow
        # 1 = Alert
        # 2 = Block
        # 3 = Monitor
        self.action_space = gym.spaces.Discrete(4)

        # State:
        # [anomaly_score, packet_rate, failed_logins, cpu_usage]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.state = None

    # Generate realistic traffic
    def generate_state(self):
        # 70% normal traffic, 30% attack
        if np.random.rand() < 0.7:
            # Normal → low anomaly
            anomaly = np.random.uniform(0.0, 0.4)
        else:
            # Attack → high anomaly
            anomaly = np.random.uniform(0.7, 1.0)

        other_features = np.random.rand(3)
        return np.array([anomaly, *other_features], dtype=np.float32)

    def reset(self):
        self.state = self.generate_state()
        return self.state

    def step(self, action):
        anomaly_score = self.state[0]

        # Ground truth
        attack = anomaly_score > 0.6

        # Reward logic
        if attack:
            if action in [1, 2, 3]:
                reward = 10      # Correct detection
            else:
                reward = -10     # Missed attack
        else:
            if action == 0:
                reward = 5       # Correct normal
            else:
                reward = -5      # False alarm

        done = False

        # Next state
        self.state = self.generate_state()

        return self.state, reward, done, {}