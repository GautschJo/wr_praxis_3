import numpy as np
import lib
import matplotlib as mpl
import matplotlib.image as mpli


####################################################################################################
# Exercise 1: Power Iteration

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    # TODO: set epsilon to default value if not set by user

    if epsilon == -1.0:
        epsilon = 10*np.finfo(float).eps

    # TODO: random vector of proper size to initialize iteration
    vector = np.random.rand(M.shape[1])

    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon

    # Perform power iteration
    while residual > epsilon:
        # TODO: implement power iteration
        ov = vector
        vector = M.dot(vector)
        vector /= np.linalg.norm(vector)
        residual = np.linalg.norm(vector - ov)
        residuals.append(residual)

    return vector, residuals


####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()
    data = []
    bilder = []
    data = lib.list_directory(path)
    for i in range(len(data)):
        if file_ending in data[i]:
            bilder.append(data[i])
    bilder.sort()
    for i in range(len(bilder)):
        x = np.array(mpl.image.imread(path + bilder[i]), dtype=np.float64)
        images.append(x)



    # TODO set dimensions according to first image in images
    dimension = np.asarray(images[0])
    dimension_y = dimension.shape[0]
    dimension_x = dimension.shape[1]

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # TODO: initialize data matrix with proper size and data type
    n = np.asarray(images[0])
    D = np.zeros((len(images), n.shape[0]*n.shape[1]), dtype=np.float64)

    # TODO: add flattened images to data matrix
    for i in range(len(images)):
        D[i] = np.ndarray.flatten(images[i])

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    # TODO: subtract mean from data / center data at origin
    mean_data = np.zeros(D.shape[1],)
    mean_data = mean_data + np.mean(D, axis=0)
    #mean_data = D - mean

    D = D - mean_data

    # TODO: compute left and right singular vectors and singular values
    # Useful functions: numpy.linalg.svd(..., full_matrices=False)

    x, svals, pcs = np.linalg.svd(D, full_matrices=False)

    #svals = np.zeros(D.shape[0])
    #pcs = np.ones_like(D)

    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    # TODO: Normalize singular value magnitudes

    k = 0
    # TODO: Determine k that first k singular values make up threshold percent of magnitude
    summe = np.sum(singular_values)
    e = summe * threshold
    d = 0
    extra = 0
    while d < e:
        d += singular_values[extra]
        extra += 1
        k += 1

    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # TODO: initialize coefficients array with proper size
    coefficients = np.zeros((len(images), pcs.shape[0]))

    # TODO: iterate over images and project each normalized image into principal component basis



    for i in range(len(images)):
        b = np.ndarray.flatten(images[i]) - mean_data
        coefficients[i] = np.dot(pcs, b)


    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """

    # TODO: load test data set
    imgs_test = []
    imgs_test, dim_test_x, dim_test_y = load_images(path_test)

    # TODO: project test data set into eigenbasis
    """
    32
    """

    coeffs_test = project_faces(pcs, imgs_test, mean_data)



    # TODO: Initialize scores matrix with proper size
    scores = np.zeros((coeffs_train.shape[0], coeffs_test.shape[0]))
    # TODO: Iterate over all images and calculate pairwise correlation
    for i in range(coeffs_train.shape[0]):
        for j in range(coeffs_test.shape[0]):
            dpro = np.dot(coeffs_train[i], coeffs_test[j])
            norm1 = np.linalg.norm(coeffs_train[i])
            norm2 = np.linalg.norm(coeffs_test[j])
            scores[i, j] = np.arccos(dpro/np.dot(norm1, norm2))




    return scores, imgs_test, coeffs_test


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
