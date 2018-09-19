import numpy as np

def train(labels, training_data):
    from features import compute_t_test, downsample_brain, crop_brain, \
        gmv_normalize_brain
    from parallelization import parallel_map

    def transform(x):
        return downsample_brain(gmv_normalize_brain(crop_brain(x)), 2)

    t_brain = compute_t_test(training_data, labels == 1, transform = transform)

    def compute_features(data, functions):
        def features(dataobj):
            return [f(dataobj) for f in functions]
        mat = np.array(parallel_map(lambda img: features(transform(img.dataobj)), data.Image))
        return [mat[:,i] for i in range(len(functions))]

    def to_feature_matrix(data):
        return np.column_stack([
            data.Age,
            (data.Gender == 'male')*1,
        ] + compute_features(data, [
            lambda x: np.sum(x * (t_brain > 3.0)),
            lambda x: np.sum(x * (t_brain < -5.0))
        ]))

    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression()
    lr_model.fit(to_feature_matrix(training_data), labels)

    def predict(test_data):
        return lr_model.predict(to_feature_matrix(test_data))

    return predict
