import pandas as pd
import numpy as np
import pickle


def load_dataset(dataset_name,
                 features,
                 scale=True,
                 test=True):
    """Load the RFD dataset from the csv file into a dataframe.

    Args:
        dataset_name (str): dataset name
        features (lst): list of features
        scale (bool): if to use scaled data (with standard scalar)
        test (bool): which split to load, train or test

    Returns:
        X: features of all instances in a dataframe
        y: labels of all instances in a numpy array
    """
    dataset_df = pd.DataFrame()

    # load rfd dataset
    if dataset_name == "mimic":
        if scale:
            if test == True:
                dataset_file_path = (
                    "rfd_model/results/combined_test_data_standardscale.csv"
                )
            else:
                dataset_file_path = (
                    "rfd_model/results/combined_training_data_standardscale.csv"
                )
        else:
            if test == True:
                dataset_file_path = ("rfd_model/results/legacy/test_data.csv")
            else:
                dataset_file_path = (
                    "rfd_model/results/legacy/training_data.csv")
        dataset_df = pd.read_csv(dataset_file_path,
                                 # header=0,
                                 engine="python")

        # deal with airway and sex not being binary, even though they should

        X = dataset_df[features]
        y = dataset_df['outcome']

    return X, y


def factual_selector(dataset, features, model, scale, seed=42, alignment='all_neg'):
    '''
    More informed choice of factual
    Can include factual with false negatives

    Args:
        dataset (str): 'mimic'
        features
        model
        seed (int): for random factual selection
        alignment (str):
            'TN' for true negatives,
            'FN' for false negatives,
            'FP' for false positives,
            'TP' for true positives,
            'all_neg' all actual negatives (FP and TN), which is the default

    Returns:
        factual: randomly selected patient

    '''
    # Load dataset
    X, y = load_dataset(dataset, features, scale=scale, test=True)

    # Make predictions on X
    pred = model.predict(X.to_numpy())

    # Compare predictions to reality
    df = pd.DataFrame(X)
    df['y_true'] = y.tolist()
    df['y_pred'] = pred.tolist()

    print(df['y_true'].value_counts())

    if alignment == 'TN':
        # all true negatives: patients who were ready for discharge and classified as such
        cases = df[(df['y_true'] == 0) & (df['y_pred'] == 0)]
    elif alignment == 'FN':
        # all false negatives: patients who were ready for discharge but not classified as such
        cases = df[(df['y_true'] == 1) & (df['y_pred'] == 0)]
    elif alignment == 'FP':
        # all false positives: patients who weren't ready for discharge and were classified as such
        cases = df[(df['y_true'] == 0) & (df['y_pred'] == 1)]
    elif alignment == 'TP':
        # all true positives: patients who were ready for discharge and were classified as such
        cases = df[(df['y_true'] == 1) & (df['y_pred'] == 1)]
    elif alignment == 'all_neg':
        # all actual negatives (FP and TN)
        cases = df[(df['y_true'] == 0)]
    else:
        print('Alignment arg not recognised, please input "FN", "FP", "TP", "TN" or "all_neg".')

    print(f'In total there are {len(cases)} {alignment} cases')

    factual = cases.sample(n=1, random_state=seed)

    print(f'Randomly selected {alignment} case: \n', factual)

    factual = factual[features]
    factual = np.array(factual)[0]

    return factual


def inverse_scaling(data, features):
    with open('rfd_model/standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    inversed = pd.DataFrame(scaler.inverse_transform(data), columns=features)

    return inversed
