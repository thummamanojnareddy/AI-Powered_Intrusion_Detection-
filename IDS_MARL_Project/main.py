from env.ids_env import IDSEnv
from agents.multi_agent_system import MultiAgentSystem

def run_system(steps=1000):
    env = IDSEnv()
    system = MultiAgentSystem()

    state = env.reset()
    print("Initial State:", state)

    correct = 0

    # Cloud statistics
    cloud_stats = {
        "AWS": {"total": 0, "attacks": 0},
        "Azure": {"total": 0, "attacks": 0},
        "GCP": {"total": 0, "attacks": 0}
    }

    for step in range(steps):
        action = system.act(state)
        next_state, reward, done, info = env.step(action)

        # Learning
        system.learn()

        # Accuracy
        if reward > 0:
            correct += 1

        cloud = info["cloud"]
        attack = info["attack"]

        cloud_stats[cloud]["total"] += 1
        if attack:
            cloud_stats[cloud]["attacks"] += 1

        print(
            f"Step {step+1} | Cloud: {cloud} | Action: {action} | Reward: {reward}"
        )

        state = next_state

    accuracy = (correct / steps) * 100
    print(f"\nSystem Accuracy: {accuracy:.2f}%")

    print("\nMulti-Cloud Traffic Summary:")
    for cloud in cloud_stats:
        print(
            f"{cloud} -> Total: {cloud_stats[cloud]['total']} | "
            f"Attacks: {cloud_stats[cloud]['attacks']}"
        )


if __name__ == "__main__":
    run_system()