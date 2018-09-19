from load_and_save import load_training_data
import model
import numpy as np

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description =
        'Cross-validate the model with training data')

    parser.add_argument('training_csv', help =
        'path to PAC2018_Covariates_Upload.csv. The training files ' +
        'PAC2018_DDDD.nii are assumed to be in the same folder. ' +
        'Notice that the format needs to be CSV instead of XLSX. ')

    return parser.parse_args()

labels, data = load_training_data(parse_arguments().training_csv)

np.random.seed(0)

# Cross-validation using repeated random subsampling
cv_scores = []
for i in range(5):
    is_training = np.random.rand(len(labels)) < 0.7
    print("test/training samples: %d/%d" % (np.sum(~is_training), np.sum(is_training)))

    training_labels = labels[is_training]
    training_data = data.loc[is_training]
    trained_model = model.train(training_labels, training_data)

    actual_test_labels = labels[~is_training]
    test_data = data.loc[~is_training]
    predicted_test_labels = trained_model(test_data)

    accuracy = np.mean(actual_test_labels == predicted_test_labels)
    print(accuracy)
    cv_scores.append(accuracy)

print('cross-validation accuracy: %.2g +- %.2g' \
    % (np.mean(cv_scores), np.std(cv_scores)))
