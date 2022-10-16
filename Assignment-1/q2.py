import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def calculate_prior(df, Y):
    """
        df: pandas dataframe
        Y: label
        Calculate prior probability of each class
        return:
            prior: prior probability
    """
    # sort the labels
    classes = sorted(list(df[Y].unique()))
    prior = []
    # calculate prior probability
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    """
        Gaussian likelihood
        df: pandas dataframe
        feat_name: feature name
        feat_val: feature value
        Y: label
        label: class label

        Calculate likelihood of a feature given a class
        return:
            p_x_given_y: likelihood
    """
    # calculate mean and standard deviation
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    # calculate likelihood
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y

def calculate_likelihood_laplace(df, feat_name, feat_val, Y, label):
    """
        Laplace likelihood
        df: pandas dataframe
        feat_name: feature name
        feat_val: feature value
        Y: label
        label: class label

        calculate likelihood of a feature given a class
        return: 
            p_x_given_y: likelihood
    """
    # calculate mean and standard deviation
    df = df[df[Y]==label]
    features = list(df.columns)[:-1]
    # calculate likelihood
    p_x_given_y = ((df[feat_name]==feat_val).sum()+1) / (len(df+1*len(features)))
    return p_x_given_y

def naive_bayes(df, X, Y,laplace_correction=False):
    """
        naive bayes classifier
        df: pandas dataframe
        X: list of features
        Y: label
        laplace_correction: boolean
        Returns:
            Y_pred: predicted labels
    """
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                if laplace_correction:
                    likelihood[j] *= calculate_likelihood_laplace(df, features[i], x[i], Y, labels[j])
                else:
                    likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])
        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))
        # print(Y_pred)

    return np.array(Y_pred) 

def split_data(df, train_ratio=0.8, test_ratio=0.2):
    """
        Split data into train and test
        df: pandas dataframe
        train_ratio: ratio of train data
        test_ratio: ratio of test data
        return:
            df1 (train): train data
            df2 (test): test data
    """
    # split the data into train and test set
    if abs(train_ratio + test_ratio - 1) > 1e-6:  # check if the sum of ratios is 1
        raise ValueError("Train and test ratio must sum to 1")
    # number of rows in train set
    train_size = int(train_ratio * len(df.index))
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
    df1 = df.iloc[:train_size, :].reset_index(drop=True)  # train set
    df2 = df.iloc[train_size:, :].reset_index(drop=True)  # test set
    return df1, df2

def get_prediction_accuracy(Y_pred, Y_test):
    """
        Calculate prediction accuracy
        Y_pred: predicted labels
        Y_test: true labels
        return:
            accuracy: prediction accuracy
    """
    accuracy = np.mean(Y_pred == Y_test)*100
    return accuracy

def Feature_Encoding(data):
    """
        Encode the features
        data: pandas dataframe
        return:
            data: pandas dataframe
    """
    # remove ID label
    data.drop('ID', axis=1, inplace=True)

    # encode the feature: Gender
    Gender_map = {gender: i for i, gender in enumerate(data['Gender'].unique())}
    data['Gender'] = data['Gender'].map(Gender_map).to_numpy()

    # encode the feature: Ever_Married
    Ever_Married_map = {married: i for i, married in enumerate(data['Ever_Married'].unique())}
    data['Ever_Married'] = data['Ever_Married'].map(Ever_Married_map).to_numpy()

    # encode the feature: Profession
    Profession_map = {ele: i for i, ele in enumerate(data['Profession'].unique())}
    data['Profession'] = data['Profession'].map(Profession_map).to_numpy()

    # encode the feature: Spending_Score
    Spending_Score_map = {ele: i for i, ele in enumerate(data['Spending_Score'].unique())}
    data['Spending_Score'] = data['Spending_Score'].map(Spending_Score_map).to_numpy()

    # encode the feature: Var_1
    Var_1_map = {ele: i for i, ele in enumerate(data['Var_1'].unique())}
    data['Var_1'] = data['Var_1'].map(Var_1_map).to_numpy()

    # encode the feature: Graduated
    Graduated_map = {ele: i for i, ele in enumerate(data['Graduated'].unique())}
    data['Graduated'] = data['Graduated'].map(Graduated_map).to_numpy()

    # encode the feature: Segmentation
    Segmentation_map = {ele: i for i, ele in enumerate(data['Segmentation'].unique())}
    data['Segmentation'] = data['Segmentation'].map(Segmentation_map).to_numpy()

    # filling Nan values with median or mode value depending on continuous or discrete feature 
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Work_Experience'].fillna(data['Work_Experience'].mode()[0], inplace=True)
    data['Family_Size'].fillna(data['Family_Size'].mode()[0], inplace=True)
    data['Var_1'].fillna(data['Var_1'].mode()[0], inplace=True)
    data['Ever_Married'].fillna(data['Ever_Married'].mode()[0], inplace=True)
    data['Spending_Score'].fillna(data['Spending_Score'].mode()[0], inplace=True)
    data['Graduated'].fillna(data['Graduated'].mode()[0], inplace=True)
    data['Profession'].fillna(data['Profession'].mode()[0], inplace=True)
    data['Segmentation'].fillna(data['Segmentation'].mode()[0], inplace=True)
    # data.fillna(data.mean(), inplace=True)
    # data.Work_Experience = data.Work_Experience.fillna(data.Work_Experience.mean())
    # data.Family_Size = data.Family_Size.fillna(data.Family_Size.mean())

    return data

def k_fold_divison(df, k=2):
    """
        Divide data into k folds
        df: pandas dataframe
        k: number of folds
        return:
            folds: list of folds
    """
    # split the data into k folds
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
    fold_size = int(len(df.index)/k)
    folds = []
    for i in range(k):
        folds.append(df.iloc[i*fold_size:(i+1)*fold_size, :].reset_index(drop=True))
    return folds

def cross_validation(df, k=2):
    """
        Perform k-fold cross validation
        df: pandas dataframe
        k: number of folds
        return:
            accuracy: list of accuracies
    """
    # split the data into k folds
    folds = k_fold_divison(df, k)
    accuracy = []
    for i in range(k):
        # get the test set
        test_set = folds[i]
        # get the train set
        train_set = pd.concat([folds[j] for j in range(k) if j!=i])
        # train the model
        Y_pred = naive_bayes(train_set, test_set.iloc[:, :-1].to_numpy(), 'Segmentation',True)
        # get the accuracy
        accuracy.append(get_prediction_accuracy(Y_pred, test_set.iloc[:, -1].to_numpy()))
    return accuracy


def Normalise_and_Outlier_Reomval(df): 
    """
        Normalise the data and remove outliers
        df: pandas dataframe
        return:
            df: pandas dataframe
    """   
    # Remove outliers from the data 
    cnt={}
    for i in range(len(df)+1):
        cnt[i]=0
    for feat in df.columns:
        if feat != 'Segmentation':
            mean = np.mean(df[feat])
            sigma = np.std(df[feat])
            upper_range = mean + 3*sigma
            # remove maximum samples outlier
            for idx in range(len(df)):
                if(df[feat][idx] > upper_range):
                    cnt[idx]+=1
    
    val_max=-1
    for idx,val in cnt.items():
        val_max=max(val,val_max)
    
    toremove=[]
    for idx,val in cnt.items():
        if val >= 0.8*val_max:
            toremove.append(idx)
    for i in range(len(toremove)):
        df.drop(toremove[i], inplace=True)

    # Normalise the data
    for feat in df.columns:
        if feat != 'Segmentation':
            # Normalise data
            df[feat] = (df[feat] - np.mean(df[feat])) / np.std(df[feat])
    return df

def main():
    """
        Main function
    """
    # read csv file
    data = pd.read_csv("Dataset_A.csv")
    
    # Feature Encoding
    data = Feature_Encoding(data)
    print('\n-----------------Feature Encoding Done-----------------\n')

    # Normalise and remove outliers
    data = Normalise_and_Outlier_Reomval(data)
    print('\n-------------------Normalising and outliers Removing Done-------------------\n')

    # print the final set of features formed
    print('\n\n-----------------Final set of features formed:-----------------\n', data.columns[:-1])

    # split the data into train and test set
    train, test = split_data(data, train_ratio=0.8, test_ratio=0.2)

    # split the test data into features and labels
    X_test = test.iloc[:,:-1].values
    Y_test = test.iloc[:,-1].values

    # split the train data into features and labels
    X_train = train.iloc[:,:-1].values
    Y_train = train.iloc[:,-1].values

    



    #### gaussian naive bayes

    print('\n\n--------------------Naive Bayes Classifier----------------------------------\n')
    
    # running on X_train 
    Y_pred = naive_bayes(train, X=X_train, Y="Segmentation")   
    print('Train accuracy: ',get_prediction_accuracy(Y_pred, Y_train))

    # running on X_test
    Y_pred = naive_bayes(train, X=X_test, Y="Segmentation")    
    print('Test accuracy: ',get_prediction_accuracy(Y_pred, Y_test))





    
    #### Naive Bayes using 10-fold cross validation
    
    print('\n\n------------------10-Fold Cross Validation------------------\n')
    accuracy = cross_validation(data, k=10)
    print('Accuracy: ', accuracy)
    print('Mean accuracy: ', np.mean(accuracy))
    print('Max accuracy: ', np.max(accuracy))




    # naive bayes using laplace correction

    print('\n\n-------------------- Using Laplace Correction -------------------\n')

    # running on X_train
    Y_pred = naive_bayes(train, X=X_train, Y="Segmentation", laplace_correction=True)   
    print('Train accuracy: ',get_prediction_accuracy(Y_pred, Y_train))

    # running on X_test
    Y_pred = naive_bayes(train, X=X_test, Y="Segmentation", laplace_correction=True)    
    print('Test accuracy: ',get_prediction_accuracy(Y_pred, Y_test))
    

if __name__ == "__main__":
    main()