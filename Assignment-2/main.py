import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
import plotly.express as px
np.random.seed(42) # for reproducibility

# figure maxopen warning ignore
import warnings
warnings.filterwarnings("ignore")

def euclidean_distance(x1, x2):
    """
        Calculates the euclidean distance between two points
        Args:
            x1: first point 
            x2: second point
    """
    return np.sqrt(np.sum((x1 - x2)**2)) # Calculate the euclidean distance

def StandardScaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        """
            Class constructor
            Args:
                K (int): number of clusters
                max_iters (int): maximum iterations to run if not converged earlier
                plot_steps (bool): whether to plot the steps of the algorithm
        """
        self.K = K # Number of clusters
        self.max_iters = max_iters # Maximum number of iterations
        self.plot_steps = plot_steps # Whether to plot the steps of the algorithm

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)] # List of clusters
        # mean feature vector for each cluster
        self.centroids = [] # List of centroids


    def predict(self, X):
        """
            Predicts cluster labels for given data samples
            Args:
                X (ndarray): data samples
        """
        self.X = X # Set the data
        self.n_samples, self.n_features = X.shape # Get the number of samples and features

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False) # Get random sample indices
        self.centroids = [self.X[idx] for idx in random_sample_idxs] # Set the centroids

        # optimize clusters
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids) # Create the clusters
            # if self.plot_steps:
            #     self.plot()

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # if self.plot_steps:
            #     self.plot()

            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

            # return cluster labels
        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        """
            Return cluster labels for each data sample
            Args:
                clusters (list): list of clusters, i.e. [cluster1, cluster2, ...]

        """
        y_pred = np.empty(self.n_samples) # Create an empty array
        for cluster_idx, cluster in enumerate(clusters): # Loop through the clusters
            for sample_idx in cluster: # Loop through the samples in the cluster
                y_pred[sample_idx] = cluster_idx # Set the cluster label
        return y_pred # Return the cluster labels

    def _create_clusters(self, centroids):
        """
            Assigns samples to closest centroids (create clusters)
            Args:
                centroids (list): list of centroids
        """
        clusters = [[] for _ in range(self.K)] # Create an empty list of clusters
        for idx, sample in enumerate(self.X): # Loop through the samples
            centroid_idx = self._closest_centroid(sample, centroids) # Get the closest centroid
            clusters[centroid_idx].append(idx) # Add the sample to the cluster
        return clusters # Return the clusters

    def _closest_centroid(self, sample, centroids):
        """
            Returns the index of the closest centroid to the sample
            Args:
                sample (ndarray): a single data sample
                centroids (list): list of centroids
        """
        distances = [euclidean_distance(sample, point) for point in centroids] # Calculate the distances
        closest_idx = np.argmin(distances) # Get the index of the closest centroid
        return closest_idx # Return the index of the closest centroid
    
    def _get_centroids(self, clusters):
        """
            Calculates new centroids as the means of the samples in each cluster
            Args:
                clusters (list): list of clusters, i.e. [cluster1, cluster2, ...]
        """
        centroids = np.zeros((self.K, self.n_features)) # Create an empty array
        for cluster_idx, cluster in enumerate(clusters): # Loop through the clusters
            cluster_mean = np.mean(self.X[cluster], axis=0) # Calculate the mean of the cluster
            centroids[cluster_idx] = cluster_mean # Set the centroid
        return centroids # Return the centroids

    def _is_converged(self, centroids_old, centroids):
        """
            Checks if clusters have converged
            Args:
                centroids_old (list): list of old centroids
                centroids (list): list of current centroids
        """
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)] # Calculate the distances
        return sum(distances) == 0 # Return True if the sum of distances is 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8)) # Create a figure and axes 
        for i, index in enumerate(self.clusters): # Loop through the clusters
            point = self.X[index].T # Get the points
            ax.scatter(*point) # Plot the points
        for point in self.centroids: # Loop through the centroids
            ax.scatter(*point, marker="x", color='black', linewidth=2) # Plot the centroids
        plt.show() # Show the plot

# normalised mutual information (NMI) is a measure of the mutual dependence between two variables
# from sklearn.metrics.cluster import normalized_mutual_info_score
def computeMI(x, y):
    """
        Computes the mutual information between two variables
        Args:
            x (ndarray): first variable
            y (ndarray): second variable
    """
    sum_mi = 0.0 # Initialize the sum of mutual information
    x_value_list = np.unique(x) # Get the unique values of x
    y_value_list = np.unique(y) # Get the unique values of y
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x) 
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in range(len(x_value_list)): # Loop through the unique values of x
        if Px[i] ==0.: # If the probability is 0
            continue
        sy = y[x == x_value_list[i]] # Get the values of y for the given x
        if len(sy)== 0: # If there are no values of y for the given x
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi

def normalized_mutual_info_score(x, y):
    """
        Computes the normalized mutual information between two variables
        Args:
            x (ndarray): first variable
            y (ndarray): second variable
    """
    mi = computeMI(x, y) # Compute the mutual information
    h_x = computeMI(x, x) # Compute the entropy of x
    h_y = computeMI(y, y) # Compute the entropy of y
    return 2.*mi/(h_x+h_y) # Return the normalized mutual information


if __name__ == "__main__":

    # Load the data
    # open file output.txt
    output_file = open("output.txt", "w")
    print("\n\n----------------------Loading data----------------------")
    df = pd.read_csv('wine.data') # Read the data
    df.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'] # Rename the columns

    print("\n\n----------------------Data----------------------", file=output_file)
    print(df.head(), file=output_file) # Print the first 5 rows of the data

    X = df.iloc[:, 1:].values # Get the features
    Y = df.iloc[:, 0].values # Get the labels

    # X_std = StandardScaler().fit_transform(X) # Standardize the features
    print("\n\n--------------------Standardizing data---------------------")
    X_std = StandardScaler(X)


    features = df.iloc[:, 1:].columns # Get the feature names
    print("\n\n----------------------Printing features----------------------")
    print(features)

    print("\n\n----------------------Instantiating PCA object with 95% variance----------------------")
    pca = PCA(0.95) # Create a PCA that will retain 95% of the variance
    # pca = PCA(n_components=2)

    print("\n\n----------------------Fitting PCA----------------------")
    principalComponents = pca.fit_transform(X_std) # Fit the PCA and transform the data

    principalDf = pd.DataFrame(data = principalComponents) # Create a new dataframe with the PCA data
    print("\n\nNumber of Components before PCA: ", X.shape[1], file=output_file)
    print("\nNumber of Components after PCA: ", principalDf.shape[1], file=output_file)

    print("\n\n----------------------Printing principal components----------------------", file=output_file)
    print(principalDf.head(), file=output_file) # Print the first 5 rows of the new data

    labels = { 
        str(i): f"PC {i+1} ({var:.1f}%)" 
        for i, var in enumerate(pca.explained_variance_ratio_ * 100) 
    } # Create a dictionary of labels for the plot

    print("\n\n----------------------Plotting principal components----------------------")
    fig = px.scatter_matrix(
        principalComponents,
        labels=labels,
        dimensions=range(principalComponents.shape[1]),
        color=df["Class"],
    ) # Create the plot
    fig.update_traces(diagonal_visible=False) # Remove the diagonal
    # fig to png
    fig.show()

    # plot all pca components w.r.t each other
    for i in range(principalComponents.shape[1]):
        for j in range(i+1,principalComponents.shape[1]):
            if i != j:
                plt.figure()
                plt.scatter(principalComponents[:, i], principalComponents[:, j], c=df["Class"])
                plt.colorbar()
                plt.xlabel(labels[str(i)])
                plt.ylabel(labels[str(j)])
                plt.title(f"PCA {i+1} vs PCA {j+1}")
                plt.savefig(f"PCA_plots/PCA{i+1}_vs_PCA{j+1}.png")

    print("\n\n----------------------Instantiating KMeans object----------------------")
    kmeans = KMeans(K=8, max_iters=150, plot_steps=True) # Create a KMeans object
    y_pred = kmeans.predict(principalComponents) # Predict the clusters
    # kmeans.plot()

    print("\n\n----------------------Printing NMI with varying values of K--------------------------", file=output_file)

    # Vary the value of K from 2 to 8. Plot the graph of K vs normalised mutual information (NMI)
    # from sklearn.metrics.cluster import normalized_mutual_info_score
    # from sklearn.cluster import KMeans
    NMI = [] # Create an empty list
    best_K = 0 # Initialize the best K
    best_NMI = 0 # Initialize the best NMI
    for i in range(2, 9): # Loop through the values of K
        kmeans = KMeans(K=i, max_iters=150, plot_steps=True) # Create a KMeans object
        # kmeans = KMeans(n_clusters=8, random_state=0).fit(principalComponents)
        y_pred = kmeans.predict(principalComponents) # Predict the clusters
        print("K = ", i, "NMI = ", normalized_mutual_info_score(Y, y_pred), file=output_file) # Print the NMI
        NMI.append(normalized_mutual_info_score(Y, y_pred)) # Append the NMI to the list
        best_K = i if normalized_mutual_info_score(Y, y_pred) > best_NMI else best_K # Update the best K
        best_NMI = normalized_mutual_info_score(Y, y_pred) if normalized_mutual_info_score(Y, y_pred) > best_NMI else best_NMI # Update the best NMI

    print("\n\n----------------------Printing best K and NMI--------------------------", file=output_file)
    print("Best K = ", best_K, "Best NMI = ", best_NMI, file=output_file) # Print the best K and NMI
    print("\n\n----------------------Plotting NMI vs K----------------------")
    plt.figure()
    plt.plot(range(2, 9), NMI)
    plt.title("NMI vs K")
    plt.xlabel("K")
    plt.ylabel("NMI")
    plt.savefig("NMI-vs-K.png")
    