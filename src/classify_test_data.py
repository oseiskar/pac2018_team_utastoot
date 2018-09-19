import numpy as np
import model
from load_and_save import load_training_data, load_test_data, save_test_predictions

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description =
        'Train the model and classify test data')

    parser.add_argument('training_csv', help =
        'Path to PAC2018_Covariates_Upload.csv. The training files ' +
        'PAC2018_DDDD.nii are assumed to be in the same folder. ' +
        'Notice that the format needs to be CSV instead of XLSX. ')

    parser.add_argument('test_csv', help =
        'path to PAC2018_Covariates_Testset.csv. The test files ' +
        'PAC2018_DDDD.nii are assumed to be in the same folder. ')

    parser.add_argument('answer_csv', help = 'Output CSV file')

    return parser.parse_args()

args = parse_arguments()

training_labels, training_data = load_training_data(args.training_csv)
test_data = load_test_data(args.test_csv)

print("test/training samples: %d/%d" % (len(training_data), len(test_data)))

trained_model = model.train(training_labels, training_data)
predicted_test_labels = trained_model(test_data)

save_test_predictions(test_data, predicted_test_labels, args.answer_csv)
