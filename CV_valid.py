import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from args_parma import get_args
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import os
import multiprocessing
from cross_validation import create_partitions_with_config


def valid(k_fold, class_type,
          train_feature, test_feature,
          train_label, test_label,
          epoch = 100):

    random_seed_list = np.random.randint(0, 10000, size=epoch)
    # create model
    if class_type == 'MLP':
        single_model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=200, max_iter=200000)
    elif class_type == 'DT':
        single_model = tree.DecisionTreeClassifier()
    elif class_type == 'LR':
        single_model = LogisticRegression(max_iter=200000)

    single_model.fit(train_feature, train_label)
    acc = accuracy_score(test_label, single_model.predict(test_feature))
    print('%s model accuracy is %f' % (class_type, acc))


    train_feature = train_feature.T
    test_feature = test_feature.T

    sum_confusion = None

    for random_seed in random_seed_list:

        # Initialize
        KF = KFold(n_splits=k_fold, shuffle=True,
                   random_state=random_seed
                   )
        model_predict = None

        for train_idx, test_idx in KF.split(train_feature):

            if class_type == 'MLP':
                class_model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=200, max_iter=200000)
            elif class_type == 'DT':
                class_model = tree.DecisionTreeClassifier()
            elif class_type == 'LR':
                class_model = LogisticRegression(max_iter=200000)

            X_feature, Y_feature = train_feature.iloc[train_idx], test_feature.iloc[train_idx]
            X_feature, Y_feature = X_feature.T, Y_feature.T
            trained_model = class_model.fit(X_feature, train_label)

            if model_predict is None:
                model_predict = trained_model.predict(Y_feature).reshape(1, -1)
            else:
                model_predict = np.concatenate((model_predict, trained_model.predict(Y_feature).reshape(1, -1)), axis=0)

        model_predict = model_predict.astype(int)

        model_confusion = None

        for row in range(model_predict.shape[0]):

            if model_confusion is None:
                model_confusion = confusion_matrix(test_label, model_predict[row, :])
            else:
                model_confusion = np.concatenate((model_confusion, confusion_matrix(test_label, model_predict[row, :])),
                                                 axis=1)

        index_predict = []

        for index in range(model_predict.shape[1]):
            index_predict.append(np.argmax(np.bincount(model_predict[:, index])))


        test_confusion = confusion_matrix(test_label, np.array(index_predict))

        model_confusion = np.concatenate((model_confusion, test_confusion), axis=1)

        if sum_confusion is None:
            sum_confusion = model_confusion

        else:
            sum_confusion = sum_confusion + model_confusion

    sum_confusion = sum_confusion / len(random_seed_list)
    test_acc = np.trace(sum_confusion[:, -class_num:]) / np.sum(sum_confusion[:, -class_num:])
    print('%s fold %s test accuracy is %f:' % (k_fold, class_type, test_acc))


    confusion_index = np.array([i + 1 for i in range(class_num)] * (k_fold + 1))

    pd_confusion = pd.DataFrame(sum_confusion)
    pd_confusion.columns = confusion_index
    return pd_confusion

if __name__ == "__main__":

    # ConV model name
    model_name = 'EfficientNetV2'

    # dataset
    data_name = 'MNIST'
    # load feature
    train_feature = pd.read_csv('path_to_feature.csv', header=None)
    test_feature = pd.read_csv('path_to_feature.csv', header=None)


    train_label = train_feature.iloc[:, 0]
    test_label = test_feature.iloc[:, 0]

    class_num = int(np.max(test_label) + 1)


    train_feature = train_feature.drop(train_feature.columns[:1], axis=1)
    test_feature = test_feature.drop(test_feature.columns[:1], axis=1)

    class_type_list = ['LR', 'DT', 'MLP']
    k_fold = 30
    for class_type in class_type_list:

        sum_confusion_matrix = valid(k_fold, class_type,
                                train_feature, test_feature,
                                train_label, test_label)

        sum_confusion_matrix.to_csv('confusion_matrix/CV/%s/%s/%s/confuse_%scv.csv' %
                                    (model_name, data_name, class_type, k_fold), header=True)
