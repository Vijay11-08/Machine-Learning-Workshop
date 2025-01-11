from mpl_toolkits.mplot3d import Axes3D

# Generate a 3D dataset
X_3d, y_3d = datasets.make_classification(n_samples=100, n_features=3, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Train an SVM classifier
clf_3d = SVC(kernel='linear')  # Linear SVM
clf_3d.fit(X_3d, y_3d)

# Plotting the 3D decision boundary
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_3d, cmap='coolwarm', marker='o')
ax.set_title('SVM in 3D (Linear)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# Create a grid to evaluate decision function
xx, yy = np.meshgrid(np.linspace(X_3d[:, 0].min(), X_3d[:, 0].max(), 30),
                     np.linspace(X_3d[:, 1].min(), X_3d[:, 1].max(), 30))
zz = np.linspace(X_3d[:, 2].min(), X_3d[:, 2].max(), 30)

# Generate decision function values
xx, yy, zz = np.meshgrid(xx, yy, zz)
decision_values = clf_3d.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
decision_values = decision_values.reshape(xx.shape)


# Plot the decision boundary
#ax.contourf(xx, yy, zz, decision_values, levels=[0], cmap='coolwarm', alpha=0.3)

plt.show()
