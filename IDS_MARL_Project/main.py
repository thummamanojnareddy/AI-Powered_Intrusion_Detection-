from env.ids_env import IDSEnv
from agents.multi_agent_system import MultiAgentSystem

def run_system(steps=1000):
    env = IDSEnv()
    system = MultiAgentSystem()

    state = env.reset()
    print("Initial State:", state)

    correct = 0

    for step in range(steps):
        action = system.act(state)
        next_state, reward, done, _ = env.step(action)

        # Learning step
        system.learn()

        if reward > 0:
            correct += 1

        print(
            f"Step {step+1} | Action: {action} | Reward: {reward} | Next State: {next_state}"
        )

        state = next_state

    accuracy = (correct / steps) * 100
    print(f"\nSystem Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_system()