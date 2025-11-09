import numpy as np
import matplotlib.pyplot as plt

# Parameters matching our rubrics
min_val = 0
max_val = 100
max_distance = max_val - min_val
decay_factor = 5.0

# Distance range
distances = np.arange(0, max_distance + 1)

# Binary reward (1 at distance 0, 0 everywhere else)
binary_rewards = np.where(distances == 0, 1.0, 0.0)

# Linear reward (matches LinearRubric)
linear_rewards = 1 - distances / max_distance

# Exponential reward (matches ExponentialRubric)
exp_rewards = np.exp(-distances / decay_factor)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(distances, binary_rewards, label='Binary: 1 if distance=0, else 0', linewidth=2.5, linestyle='--')
plt.plot(distances, linear_rewards, label=f'Linear: 1 - distance/{max_distance}', linewidth=2)
plt.plot(distances, exp_rewards, label=f'Exponential: exp(-distance/{decay_factor})', linewidth=2)

plt.xlabel('Distance from Target')
plt.ylabel('Reward')
plt.title('Reward Gradients: Binary vs Linear vs Exponential')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, max_distance)
plt.ylim(-0.05, 1.1)

plt.savefig('reward_gradient.png', dpi=300, bbox_inches='tight')
print("Saved reward_gradient.png")
