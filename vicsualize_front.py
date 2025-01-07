import pandas as pd
import matplotlib.pyplot as plt

# load data
file_path = 'results/new_pareto_nsga2_data.csv'
data = pd.read_csv(file_path)

# Set the profit column to a negative value
data['profit'] = -data['profit']

# Drawing all frontiers, using different colours
plt.figure(figsize=(10, 6))

# Scatterplot each frontier by grouping of fronts
for front_value in data['front'].unique():
    front_data = data[data['front'] == front_value]
    plt.scatter(front_data['time'], front_data['profit'], label=f'Front {front_value}', alpha=0.7)

# Add tags and titles
plt.xlabel('Time')
plt.ylabel('Profit (Negative)')
plt.title('All Fronts: Profit vs. Time (Negative Profit)')
plt.legend(title='Pareto Front')

# Show charts
plt.grid(True)
plt.show()