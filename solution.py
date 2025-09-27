import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):

    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()

def find_nearest_center(X, Centers):

    # Initialize return array, an array of X.shape() which stores indices of a cluster that each object of X belongs to
    clusters_indices = []

    for x in X:
        # Calculate the distance from the point 'x' to ALL centers at once.
        distances = np.linalg.norm(Centers - x, axis=1)

        # Find nearest center and append its index to the return array
        nearest_center_index = int(np.argmin(distances))

        clusters_indices.append(nearest_center_index)


    return clusters_indices

def calculate_new_centers(X, clusters_indices):

    # Initialize the array for storing clusters of X's features
    k = np.max(clusters_indices) + 1
    clusters = [[] for _ in range(k)]

    # Create arrays of clusters storing features belonging to them
    for i in range(len(X)):
        clusters[clusters_indices[i]].append(X[i])

    # Return the coordinates of new centers
    return np.array([np.mean(cluster,axis=0) for cluster in clusters])

class CustomKMeans:
    def __init__(self, k):

       self.k = k
       self.centers = None

    def fit(self, X, eps=1e-6):
        # Initialize centers and helper variable - old centers
        current_centers = np.array([X[i] for i in range(self.k)])
        old_centers = np.zeros_like(current_centers)

        while np.linalg.norm(current_centers - old_centers) > eps:
            # Assign current center to the old one
            old_centers = current_centers.copy()

            # Helper variable for updating the centers
            assignments = find_nearest_center(X, old_centers)

            # Update the current centers
            current_centers = calculate_new_centers(X, assignments)

        # Pass final centers to the self variable
        self.centers = current_centers

    def predict(self, X):
       return find_nearest_center(X, self.centers)

if __name__ == '__main__':

    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    Centers = [X_full[0], X_full[1], X_full[2]]

    # Result of Stage 1
    #print(find_nearest_center(X_full, Centers))

    # Result of Stage 2 - flatten the result to 1-D array as in the objective
    #print(calculate_new_centers(X_full, Centers).flatten().tolist())

    # Result of Stage 3
    custom_k_means = CustomKMeans(k=2)
    custom_k_means.fit(X_full)
    predicted_labels = custom_k_means.predict(X_full[:10])
    print(predicted_labels)
