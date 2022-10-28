"""
Credits: This code is borrowed from https://github.com/Microsoft/FERPlus/tree/master/src
"""

import sys
import os
import csv
import numpy as np
from PIL import Image
from boundingBoxUtil import Rect
from imageUtils import compute_norm_mat, distort_img, preproc_img


def display_summary(train_data_reader, val_data_reader, test_data_reader):
    emotion_count = train_data_reader.emotion_count
    emotion_header = ['happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
    print("{0}\t{1}\t{2}\t{3}".format("".ljust(10), "Train", "Val", "Test"))
    for index in range(emotion_count):
        print("{0}\t{1}\t{2}\t{3}".format(emotion_header[index].ljust(10),
                                          train_data_reader.per_emotion_count[index],
                                          val_data_reader.per_emotion_count[index],
                                          test_data_reader.per_emotion_count[index]))


class FERPlusParameters:
    def __init__(self, target_size, width, height, training_mode="majority", deterministic=False, shuffle=True):
        self.target_size = target_size
        self.width = width
        self.height = height
        self.training_mode = training_mode
        self.deterministic = deterministic
        self.shuffle = shuffle

class FERPlusReader(object):
    @classmethod
    def create(cls, base_folder, sub_folders, label_file_name, parameters):
        reader = cls(base_folder, sub_folders, label_file_name, parameters)
        reader.load_folders(parameters.training_mode)
        return reader

    def __init__(self, base_folder, sub_folders, label_file_name, parameters):
        '''
        Each sub_folder contains the image files and a csv file for the corresponding label. The read iterate through
        all the sub_folders and aggregate all the images and their corresponding labels.
        '''
        self.base_folder = base_folder
        self.sub_folders = sub_folders
        self.label_file_name = label_file_name
        self.emotion_count = parameters.target_size
        self.width = parameters.width
        self.height = parameters.height
        self.shuffle = parameters.shuffle
        self.training_mode = parameters.training_mode

        # data augmentation parameters.deterministic
        if parameters.deterministic:
            self.max_shift = 0.0
            self.max_scale = 1.0
            self.max_angle = 0.0
            self.max_skew = 0.0
            self.do_flip = False
        else:
            self.max_shift = 0.08
            self.max_scale = 1.05
            self.max_angle = 20.0
            self.max_skew = 0.05
            self.do_flip = True

        self.data = None
        self.per_emotion_count = None
        self.batch_start = 0
        self.indices = 0

        self.A, self.A_pinv = compute_norm_mat(self.width, self.height)

    def has_more(self):
        '''
        Return True if there is more min-batches.
        '''
        if self.batch_start < len(self.data):
            return True
        return False

    def reset(self):
        '''
        Start from beginning for the new epoch.
        '''
        self.batch_start = 0

    def size(self):
        '''
        Return the number of images read by this reader.
        '''
        return len(self.data)

    def process_images(self):
        '''
        Return the next mini-batch, we do data augmentation during constructing each mini-batch.
        '''
        data_size = len(self.data)
        inputs = np.empty(shape=(data_size, 1, self.width, self.height), dtype=np.float32)
        targets = np.empty(shape=(data_size, self.emotion_count), dtype=np.float32)

        for idx in range(data_size):
            index = self.indices[idx]
            distorted_image = distort_img(self.data[index][1],
                                          self.data[index][3],
                                          self.width,
                                          self.height,
                                          self.max_shift,
                                          self.max_scale,
                                          self.max_angle,
                                          self.max_skew,
                                          self.do_flip)
            final_image = preproc_img(distorted_image, A=self.A, A_pinv=self.A_pinv)

            inputs[idx - self.batch_start] = final_image
            targets[idx - self.batch_start, :] = self._process_target(self.data[index][2])

        return inputs, targets

    def load_folders(self, mode):
        '''
        Load the actual images from disk. While loading, we normalize the input data.
        '''
        self.reset()
        self.data = []
        self.per_emotion_count = np.zeros(self.emotion_count, dtype=np.int)

        for folder_name in self.sub_folders:
            print("Loading %s" % (os.path.join(self.base_folder, folder_name)))
            folder_path = os.path.join(self.base_folder, folder_name)
            in_label_path = os.path.join(folder_path, self.label_file_name)
            with open(in_label_path) as csvfile:
                emotion_label = csv.reader(csvfile)
                for row in emotion_label:
                    # load the image
                    image_path = os.path.join(folder_path, row[0])
                    image_data = Image.open(image_path)
                    image_data.load()

                    # face rectangle
                    box = list(map(int, row[1][1:-1].split(',')))
                    face_rc = Rect(box)

                    emotion_raw = list(map(float, row[2:len(row)]))
                    emotion = self._process_data(emotion_raw, mode)
                    idx = np.argmax(emotion)
                    if idx < self.emotion_count:  # not unknown or non-face
                        emotion = emotion[:-2]
                        emotion = [float(i) / sum(emotion) for i in emotion]
                        self.data.append((image_path, image_data, emotion, face_rc))
                        self.per_emotion_count[idx] += 1

        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _process_target(self, target):
        '''
        Based on https://arxiv.org/abs/1608.01041 the target depend on the training mode.
        Majority or crossentropy: return the probability distribution generated by "_process_data"
        Probability: pick one emotion based on the probability distribtuion.
        Multi-target:
        '''
        if self.training_mode == 'majority' or self.training_mode == 'crossentropy':
            return target
        elif self.training_mode == 'probability':
            idx = np.random.choice(len(target), p=target)
            new_target = np.zeros_like(target)
            new_target[idx] = 1.0
            return new_target
        elif self.training_mode == 'multi_target':
            new_target = np.array(target)
            new_target[new_target > 0] = 1.0
            epsilon = 0.001  # add small epsilon in order to avoid ill-conditioned computation
            return (1 - epsilon) * new_target + epsilon * np.ones_like(target)

    def _process_data(self, emotion_raw, mode):
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0
        sum_list = sum(emotion_raw)
        emotion = [0.0] * size
        if mode == 'majority':
            maxval = max(emotion_raw)
            if maxval > 0.5 * sum_list:
                emotion[np.argmax(emotion_raw)] = maxval
            else:
                emotion = emotion_unknown
        return [float(i) / sum(emotion) for i in emotion]
