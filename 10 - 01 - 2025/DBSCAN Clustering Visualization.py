import matplotlib.pyplot as plt

# Plotting the points with different colors for Core, Border, and Noise
colors = {'Core': 'blue', 'Border': 'green', 'Noise': 'red'}
plt.figure(figsize=(10, 8))

for i, (x, y) in enumerate(points):
    plt.scatter(x, y, color=colors[point_types[i]], label=point_types[i] if point_types[i] not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(x + 0.1, y, f"P{i+1}", fontsize=10)  # Annotate points

# Add plot details
plt.title("DBSCAN Clustering Visualization", fontsize=14)
plt.xlabel("X Coordinate", fontsize=12)
plt.ylabel("Y Coordinate", fontsize=12)
plt.grid(True)
plt.legend(title="Point Type")
plt.show()
