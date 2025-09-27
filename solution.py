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

    # Initialize return array, an array of X.shape() which stores indeces of a cluster that each object of X belongs to
    clusters_indeces = []

    for x in X:
        # Euclidean distances for each of the centers and -i-1th points of X
        d1 = np.sqrt(np.sum(np.square(Centers[0]-x)))
        d2 = np.sqrt(np.sum(np.square(Centers[1]-x)))
        d3 = np.sqrt(np.sum(np.square(Centers[2]-x)))

        # Find nearest center and append its index to the return array
        center = min(d1,d2,d3)
        if center == d1:
            clusters_indeces.append(0)
        elif center == d2:
            clusters_indeces.append(1)
        elif center == d3:
            clusters_indeces.append(2)

    return clusters_indeces

def calculate_new_centers(X, Centers):

    # Initialize the array for storing clusters of X's features
    clusters = [[],[],[]]
    # Calculate the which initial center is closest to each feature
    clusters_indeces = find_nearest_center(X, Centers)
    
    # Create arrays of clusters storing features belonging to them
    for i in range(len(X)):
        clusters[clusters_indeces[i]].append(X[i])

    # Return the coordinates of new centers
    return np.array([np.mean(cluster,axis=0) for cluster in clusters])

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
    print(calculate_new_centers(X_full, Centers).flatten().tolist())
