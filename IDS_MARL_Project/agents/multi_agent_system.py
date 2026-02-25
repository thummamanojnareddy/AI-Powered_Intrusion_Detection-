from agents.dqn_agent import DQNAgent

class MultiAgentSystem:
    def __init__(self, num_agents=3):
        self.agents = [DQNAgent() for _ in range(num_agents)]

    def act(self, state):
        actions = [agent.act(state) for agent in self.agents]
        return max(set(actions), key=actions.count)

    def learn(self):
        for agent in self.agents:
            agent.learn()