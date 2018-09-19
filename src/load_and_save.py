import os
import numpy as np
import pandas as pd
import nibabel as nib

def add_image_data(meta, data_path):
    data = []
    for name in meta.PAC_ID:
        img = nib.load(os.path.join(data_path, name+".nii"))
        data.append(img)
    return meta.assign(Image=data)

def load_metadata(filename):
    meta = pd.read_csv(filename)
    meta['Gender'] = meta.Gender.map({1: 'male', 2: 'female'})
    return meta

def get_labels_and_features(meta):
    # convert labels 1 and 2 to more standard 0 and 1
    labels = meta.Label.values - 1
    features = meta.drop(['Label'], axis=1)
    return (labels, features)

def load_training_data(meta_filename):
    meta = load_metadata(meta_filename)

    # automatically find correct path, data/ or ../data/ where
    # the data files are located
    data_path = os.path.dirname(meta_filename)
    data = add_image_data(meta, data_path)

    labels, features = get_labels_and_features(data)
    return (labels, features)

def load_test_data(meta_filename):
    meta = load_metadata(meta_filename)
    data_path = os.path.dirname(meta_filename)
    return add_image_data(meta, data_path)

def save_test_predictions(test_data, predicted_test_labels, answer_file):
    df = pd.DataFrame(
        data={'Label': np.round(predicted_test_labels + 1)},
        index=test_data.PAC_ID)
    df.to_csv(answer_file)
