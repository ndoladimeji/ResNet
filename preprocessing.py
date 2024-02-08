
if __name__ == "__main__":
    # import pandas
    import pandas as pd
    # read the JustRAIGS_Train_labels csv file into a pandas dataframe
    df = pd.read_csv('JustRAIGS_Train_labels.csv', sep=';')
    # Save all rows and the first four columns to a new dataframe
    new_df = df.iloc[:, :4]
    new_df
    # create a column for target variable from Final Label column
    # replace 'NRG' with numerical zero meaning non referrable glaucoma
    # replace 'RG' with numerical one meaning referrable glaucoma
    new_df['target'] = new_df['Final Label']
    new_df['target'] = new_df['target'].replace(['NRG'], 0)
    new_df['target'] = new_df['target'].replace(['RG'], 1)
    new_df['target'].nunique()
    new_df['target'].value_counts()
    # save the new dataframe to a csv file
    new_df.to_csv("Rtrain.csv", index=False)

    # the data is skewed; more than 80% cases are non referrable glaucoma
    # create training folds to increase randomness of the training set
    from sklearn.model_selection import StratifiedKFold

    # Create a new column 'kfold' and initialize it with -1
    new_df["kfold"] = -1

    # Randomize the training data
    new_df = new_df.sample(frac=1).reset_index(drop=True)

    # Get the target variable from the DataFrame
    y = new_df.target.values

    # Create an instance of StratifiedKFold
    kf = StratifiedKFold(n_splits=5)

    # Assign the fold index to the 'kfold' column for each row
    for fold_, (_, valid_index) in enumerate(kf.split(X=new_df, y=y)):
        new_df.loc[valid_index, "kfold"] = fold_

    # Save the training data with fold indices to a CSV file
    new_df.to_csv('Rtrain_folds.csv', index=False)