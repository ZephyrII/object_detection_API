# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

import PIL.Image
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                    'merged set.')

flags.DEFINE_string('samples_per_file', '200',
                    'Samples per tfrecord file')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def dict_to_tf_example(xml_data, img_fname, label_fname):

    with tf.gfile.GFile(img_fname, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format == "PNG":
        image = image.convert('RGB')
    # if image.format != 'JPEG':
    #     raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    print(label_fname)
    label = cv2.imread(label_fname)
    width = image.width
    height = image.height
    mask_coords = np.argwhere(label == 1)
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_fname.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(img_fname.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/channels': dataset_util.int64_feature(3),
        'image/shape': dataset_util.int64_list_feature([height, width, 3]),
    }
    if mask_coords.shape[0] > 0:

        obj = xml_data.find('object')
        rel_xmin = np.min(mask_coords[:, 1])
        rel_ymin = np.min(mask_coords[:, 0])
        rel_xmax = np.max(mask_coords[:, 1])
        rel_ymax = np.max(mask_coords[:, 0])
        xmin = [rel_xmin/width]
        ymin = [rel_ymin/height]
        xmax = [rel_xmax/width]
        ymax = [rel_ymax/height]


        keypoints_x = []
        keypoints_y = []
        for i in range(6):
            kp_xml = obj.find('keypoints').find('keypoint'+str(i))
            keypoints_x.append((float(kp_xml.find('x').text) - rel_xmin) / (rel_xmax - rel_xmin))
            keypoints_y.append((float(kp_xml.find('y').text) - rel_ymin) / (rel_ymax - rel_ymin))
        # for kp_xml in obj.find('keypoints').findall('keypoint'):
        #     keypoints_x.append((float(kp_xml.find('x').text)-rel_xmin)/(rel_xmax-rel_xmin))
        #     keypoints_y.append((float(kp_xml.find('y').text)-rel_ymin)/(rel_ymax-rel_ymin))

        class_name = obj.find('name').text
        classes = []
        classes_text = [class_name.encode('utf8')]
        if class_name=='charger':
            classes.append(1)
        else:
            classes.append(2)


        feature_dict['image/object/keypoint/x'] = dataset_util.float_list_feature(keypoints_x)
        feature_dict['image/object/keypoint/y'] = dataset_util.float_list_feature(keypoints_y)
        feature_dict['image/object/bbox/xmin'] = dataset_util.float_list_feature(xmin)
        feature_dict['image/object/bbox/xmin'] = dataset_util.float_list_feature(xmin)
        feature_dict['image/object/bbox/xmax'] = dataset_util.float_list_feature(xmax)
        feature_dict['image/object/bbox/ymin'] = dataset_util.float_list_feature(ymin)
        feature_dict['image/object/bbox/ymax'] = dataset_util.float_list_feature(ymax)
        feature_dict['image/object/class/text'] = dataset_util.bytes_list_feature(classes_text)
        feature_dict['image/object/class/label'] = dataset_util.int64_list_feature(classes)

        encoded_mask_png_list = []
        pil_image = PIL.Image.fromarray(label)
        output_io = io.BytesIO()
        pil_image.save(output_io, format='PNG')
        encoded_mask_png_list.append(output_io.getvalue())
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def get_output_filename(output_dir, idx):
    # if idx % 10 == 1:
    #     return '%s/charger_test_%03d.tfrecord' % (output_dir, idx)
    # else:
    return '%s/charger_train_%03d.tfrecord' % (output_dir, int(idx))


def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    print(label_map_dict)

    im_path = os.path.join(data_dir, "images")
    labels_path = os.path.join(data_dir, "labels")
    annotations_path = os.path.join(data_dir, "annotations")
    idx = 0
    tf_idx = 0
    file_list = os.listdir(im_path)
    print(im_path)
    # file_list = shuffle(file_list)
    while idx < len(file_list):
        tf_filename = get_output_filename(FLAGS.output_path, tf_idx)
        writer = tf.python_io.TFRecordWriter(tf_filename)
        j = 0
        while j < int(FLAGS.samples_per_file):
            fname = file_list[idx]
            img_full_path = os.path.join(im_path, fname)
            label_full_path = os.path.join(labels_path, fname[:-4] + '_label' + fname[-4:])
            annotations_full_path = os.path.join(annotations_path, fname[:-4] + '.txt')
            xml_data = None
            try:
                tree = ET.parse(annotations_full_path)
                xml_data = tree.getroot()
            except:
                print("No annotation file. Adding negatve sample")

            tf_example = dict_to_tf_example(xml_data, img_full_path, label_full_path)
            writer.write(tf_example.SerializeToString())
            idx += 1
            j += 1
        tf_idx += 1
        writer.close()


if __name__ == '__main__':
    tf.app.run()
