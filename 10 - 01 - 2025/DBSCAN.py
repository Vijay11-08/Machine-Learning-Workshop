# Update the "Neighbors" column to use point identifiers (e.g., P1, P2) instead of indices
data_table_with_points = {
    "Point": [f"P{i+1}" for i in range(len(points))],
    "Coordinates": [tuple(p) for p in points],
    "Neighbors (Points)": [[f"P{j+1}" for j in n] for n in neighbors],
    "Neighbors (Count)": [len(n) for n in neighbors],
    "Type": point_types
}


# Convert to DataFrame for better visualization
data_table_with_points_df = pd.DataFrame(data_table_with_points)
data_table_with_points_df
