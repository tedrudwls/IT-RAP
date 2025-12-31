import re
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Define reward extraction function ===
def extract_rewards_from_file(file_path):
    rewards = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"Episode\s+\d+:\s+([0-9.]+)", line)
            if match:
                reward = float(match.group(1))
                rewards.append(reward)
    return rewards

# === 2. List of file paths ===
file_paths = [
    # ("C:\\Users\\seolp\\Desktop\\1.txt", "CelebA 100 images | StarGAN", "tab:orange"), # 1 
    # ("C:\\Users\\seolp\\Desktop\\3.txt", "CelebA 100 images | AttGAN", "tab:blue"), # 3
    ("C:\\Users\\Bandi\\Desktop\\stargan & attgan\\result_test\\[stargan]reward_moving_avg.txt", "MAAD 100 images | StarGAN", "tab:orange"), # 2
    ("C:\\Users\\Bandi\\Desktop\\stargan & attgan\\result_test\\[attgan]reward_moving_avg.txt", "MAAD 100 images | AttGAN", "tab:blue") # 4
]

window_size = 50
plt.figure(figsize=(14, 6))

# === 3. Extract rewards, normalize, and visualize moving average for each file ===
for path, label, color in file_paths:
    rewards = extract_rewards_from_file(path)
    series = pd.Series(rewards)

    # Z-score normalization
    mean = series.mean()
    std = series.std()
    normalized = (series - mean) / std

    # Centered moving average
    smoothed = normalized.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Visualization
    plt.plot(smoothed.index, smoothed.values, label=f"{label}", color=color, linewidth=2)


# === 4. Customize the plot ===
plt.title("Reward Trend Over Episodes", fontsize=20)
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Normalized Reward", fontsize=20)
# plt.ylim(10, 60)              
# plt.ylim(40, 45) -> Suitable when running only 1.txt
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc='upper left', fontsize=18)
plt.tight_layout()
plt.savefig("reward_trend.png")
plt.show()