import tensorflow as tf
import os
import numpy as np

def inspect_tfrecord(file_path):
    print("Checking file: {}".format(file_path))
    
    if not os.path.exists(file_path):
        print("File not found: {}".format(file_path))
        return

    # TFRecord iterator (compatible with TF 1.x and 2.x)
    try:
        record_iterator = tf.python_io.tf_record_iterator(path=file_path)
    except AttributeError:
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=file_path)

    count = 0
    all_examples = []
    all_labels = []

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        feature_map = example.features.feature

        if 'example' in feature_map:
            all_examples.append(list(feature_map['example'].float_list.value))
        if 'label' in feature_map:
            all_labels.append(list(feature_map['label'].float_list.value))

        # Print details only for the first record
        if count == 0:
            print("First record structure:")
            feature_map = example.features.feature
            for key in feature_map:
                feature = feature_map[key]
                kind = feature.WhichOneof('kind')
                
                if kind == 'float_list':
                    values = feature.float_list.value
                    print("  Key: '{}'".format(key))
                    print("    Type: float_list")
                    print("    Length: {}".format(len(values)))
                    print("    First 5 values: {}".format(values[:5]))
                elif kind == 'int64_list':
                    values = feature.int64_list.value
                    print("  Key: '{}'".format(key))
                    print("    Type: int64_list")
                    print("    Length: {}".format(len(values)))
                    print("    First 5 values: {}".format(values[:5]))
                elif kind == 'bytes_list':
                    values = feature.bytes_list.value
                    print("  Key: '{}'".format(key))
                    print("    Type: bytes_list")
                    print("    Length: {}".format(len(values)))
        
        count += 1

    print("Total records: {}".format(count))

    # Save to numpy
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)

    save_dict = {}
    if all_examples:
        np_examples = np.array(all_examples, dtype=np.float32)
        save_dict['example'] = np_examples
        print("Examples shape: {}".format(np_examples.shape))
    if all_labels:
        np_labels = np.array(all_labels, dtype=np.float32)
        save_dict['label'] = np_labels
        print("Labels shape: {}".format(np_labels.shape))

    if save_dict:
        save_path = os.path.join(dir_name, "{}.npz".format(base_name))
        np.savez(save_path, **save_dict)
        print("Saved to: {}".format(save_path))

    print("-" * 40)

if __name__ == "__main__":
    # Check in 'wifi' directory by default as per STFNets.py, or current dir
    target_files = ['train.tfrecord', 'eval.tfrecord']
    search_dirs = ['wifi', 'hhar', '.']
    
    for filename in target_files:
        found = False
        for d in search_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                inspect_tfrecord(path)
                found = True
                break
        if not found:
            print("Could not find {} in {}".format(filename, search_dirs))
