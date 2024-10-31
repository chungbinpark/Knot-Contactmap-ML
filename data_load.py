# data_load.py
import numpy as np
import random

def load_data(ns1, ns2, ns3):
    type_list = ['un', '3_1']
    
    # Load train, test, and validation data
    train_images = np.concatenate(
        [np.load(f'./augmented_data/{type}/train_image.npy') for type in type_list], axis=0
    )
    test_images = np.concatenate(
        [np.load(f'./augmented_data/{type}/test_image.npy') for type in type_list], axis=0
    )
    valid_images = np.concatenate(
        [np.load(f'./augmented_data/{type}/valid_image.npy') for type in type_list], axis=0
    )
    
    # Shuffle train data
    train_labels = np.concatenate((np.ones(ns1), np.zeros(ns1)))  # 예시로 1과 0으로 라벨링
    test_labels = np.concatenate((np.ones(ns2), np.zeros(ns2)))
    valid_labels = np.concatenate((np.ones(ns3), np.zeros(ns3)))

    def shuffle_data(images, labels):
        tmp = list(zip(images, labels))
        random.shuffle(tmp)
        shuffled_images, shuffled_labels = zip(*tmp)
        return np.array(shuffled_images), np.array(shuffled_labels)

    train_images, train_labels = shuffle_data(train_images, train_labels)
    test_images, test_labels = shuffle_data(test_images, test_labels)
    valid_images, valid_labels = shuffle_data(valid_images, valid_labels)

    return train_images, train_labels, test_images, test_labels, valid_images, valid_labels

