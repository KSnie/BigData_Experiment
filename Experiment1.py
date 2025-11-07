import matplotlib.pyplot as plt  
import matplotlib.image as img 
import os
import numpy as np

def create_database(train_database_path='Experiment-1/TrainDatabase/'):
    train_number = 0
    ls = os.listdir(train_database_path)
    for i in ls:
        if os.path.isfile(os.path.join(train_database_path, i)):
            train_number += 1

    temp_array = []
    for i in range(1, train_number + 1):
        train_image_path = 'Experiment-1/TrainDatabase/' + str(i) + '.jpg'
        train_image = img.imread(train_image_path)
        one_d_train_image = np.reshape(np.array(train_image), (-1, 1), order="F")
        temp_array.append(one_d_train_image)
    one_d_train_image_all_set = np.transpose(np.asmatrix(np.array(temp_array)))
    return one_d_train_image_all_set

def eigen_face_core(one_d_train_image_all_set):
    mean_of_train_database = np.asmatrix(np.mean(one_d_train_image_all_set, 1))
    train_number = one_d_train_image_all_set.shape[1]
    centered_image_vectors_temp = []
    for i in range(train_number):
        temp = one_d_train_image_all_set[:, i] - mean_of_train_database
        centered_image_vectors_temp.append(temp)
    centered_image_vectors = np.transpose(np.asmatrix(np.array(centered_image_vectors_temp)))
    covariance_matrix_temp = np.dot(np.transpose(centered_image_vectors), centered_image_vectors)
    eigenvalues, feature_vector = np.linalg.eig(covariance_matrix_temp)
    eigen_vector = []
    for i in range(feature_vector.shape[1]):
        if eigenvalues[i] > 1:
            eigen_vector.append(feature_vector[:, i])
    eigen_vector = np.transpose(np.asmatrix(np.array(eigen_vector)))
    eigen_faces = np.dot(centered_image_vectors, eigen_vector)

    return mean_of_train_database, eigen_faces, centered_image_vectors

def recognition(test_image, mean_of_train_database, eigen_faces, centered_image_vectors):
    projected_image = []
    train_number = eigen_faces.shape[1]
    for i in range(train_number):
        temp = np.dot(np.transpose(eigen_faces), centered_image_vectors[:, i])
        projected_image.append(temp)
    projected_image = np.transpose(np.asmatrix(np.array(projected_image)))

    in_image = np.reshape(np.array(test_image), (-1, 1), order="F")
    difference = in_image - mean_of_train_database
    projected_test_image = np.dot(np.transpose(eigen_faces), difference)

    euclidean_distance = []
    for i in range(train_number):
        q = projected_image[:, i]
        temp = np.linalg.norm(projected_test_image - q) ** 2
        euclidean_distance.append(temp)
    euclidean_distance = np.transpose(np.asmatrix(np.array(euclidean_distance)))

    euclidean_distance = np.array(euclidean_distance)
    euclidean_distance_min_index = np.argmin(euclidean_distance) + 1

    output_name = str(euclidean_distance_min_index) + '.jpg'
    return output_name

def main():
    image_order = input('Enter test image name (a number between 1 to 10, q to quit): ')
    if image_order == 'q':
        quit()
    num = int(image_order)
    while num < 1 or num > 10:
        print('Illegal input,please input again')
        image_order = input('Enter test image name (a number between 1 to 10, q to quit): ')
        if image_order == 'q':
            quit()
        num = int(image_order)
    test_image_path = 'Experiment-1/TestDatabase/' + str(image_order) + '.jpg'
    test_image = img.imread(test_image_path)

    plt.imshow(test_image)
    plt.axis('off')
    plt.title('Input')
    plt.show()

    one_d_train_image_all_set = create_database()
    mean_of_train_database, eigen_faces, centered_image_vectors = eigen_face_core(one_d_train_image_all_set)
    output_name = recognition(test_image, mean_of_train_database, eigen_faces, centered_image_vectors)

    selected_image = 'Experiment-1/TrainDatabase/' + output_name
    selected_image = img.imread(selected_image)

    plt.imshow(selected_image)
    plt.axis('off')
    plt.title('Equivalent Image')
plt.show()

if __name__ == '__main__':
    while 1:
        main()