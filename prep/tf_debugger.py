
import tensorflow as tf


def tfrecord_debugger(fn):
    """
    This is to debug your tf records, for sanity check
    :param fn: tf record
    :return: printing of shape of the tf record file elements
    """
    def tfrec_data_input_fn(filenames, batch_size=64, shuffle=False):
        def _input_fn():
            def _parse_record(tf_record):
                features = {
                    'crop': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([], dtype=tf.int64),
                    'height': tf.FixedLenFeature([], dtype=tf.int64),
                    'width': tf.FixedLenFeature([], dtype=tf.int64),
                    'depth': tf.FixedLenFeature([], dtype=tf.int64)
                }
                record = tf.parse_single_example(tf_record, features)

                image_raw = tf.decode_raw(record['crop'], tf.uint8)
                height = tf.cast(record['height'], tf.int32)
                width = tf.cast(record['width'], tf.int32)
                depth = tf.cast(record['depth'], tf.int32)
                image_raw = tf.reshape(image_raw, shape=(height, width, depth))

                label = tf.cast(record['label'], tf.int32)
                label = tf.one_hot(label, depth=7)

                return image_raw, label
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(_parse_record)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat(3)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            # X = {'image': features}
            # y = labels
            # return X, y
            return features, labels
        return _input_fn
    #
    tfrec_dev_input_fn = tfrec_data_input_fn([fn], batch_size=64)
    features, labels = tfrec_dev_input_fn()
    with tf.Session() as sess:
        img, label = sess.run([features, labels])
        print("are these ones the correct dimensions? {}-{}".format(img.shape, label.shape))
    sess.close()
    return None


if __name__ == '__main__':
    fn = '/Volumes/SSD_ML/DATA/ML/FER_DATASET/fer2013/20190910_facialexpr_fer2013_asaformat/training/train_006-010.tfrecord'
    tfrecord_debugger(fn)
    pass