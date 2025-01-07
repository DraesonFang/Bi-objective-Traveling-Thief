import pandas as pd
import matplotlib.pyplot as plt

# Create the data
data = {
    'Generations': [10, 20, 30, 40, 50],
    'GA_Time': [365772.94, 365772.87, 365593.29, 364993.33, 364971.10],
    'NSGA_II_Time': [113215.88, 109168.10, 89274.23, 79804.74, 63980.08],
    'GA_Profit': [23512.0, 23712.0, 25212.0, 25992.0, 26430.0],
    'NSGA_II_Profit': [27686.0, 31049.0, 34556.0, 37793.0, 41815.0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Time Comparison
ax1.plot(df['Generations'], df['GA_Time'], 'ro-', label='GA', linewidth=2, markersize=8)
ax1.plot(df['Generations'], df['NSGA_II_Time'], 'bs-', label='NSGA-II', linewidth=2, markersize=8)
ax1.set_xlabel('Generations')
ax1.set_ylabel('Time')
ax1.set_title('Time Comparison: GA vs NSGA-II')
ax1.legend()
ax1.grid(True)

# Plot Profit Comparison
ax2.plot(df['Generations'], df['GA_Profit'], 'ro-', label='GA', linewidth=2, markersize=8)
ax2.plot(df['Generations'], df['NSGA_II_Profit'], 'bs-', label='NSGA-II', linewidth=2, markersize=8)
ax2.set_xlabel('Generations')
ax2.set_ylabel('Profit')
ax2.set_title('Profit Comparison: GA vs NSGA-II')
ax2.legend()
ax2.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Print improvement percentages
print("\nPerformance Improvement Analysis:")
final_time_improvement = ((df['GA_Time'].iloc[-1] - df['NSGA_II_Time'].iloc[-1]) / df['GA_Time'].iloc[-1]) * 100
final_profit_improvement = ((df['NSGA_II_Profit'].iloc[-1] - df['GA_Profit'].iloc[-1]) / df['GA_Profit'].iloc[-1]) * 100

print(f"Time Reduction: {final_time_improvement:.2f}%")
print(f"Profit Increase: {final_profit_improvement:.2f}%")