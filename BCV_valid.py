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
from joblib import Parallel, delayed
import time
# args = get_args()

# 模型


def clf_parallel(class_type, train_label, train_feature, test_feature, train_idx, valid_idx):

    if class_type == 'MLP':
        class_model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=50, max_iter=200000)
    elif class_type == 'DT':
        class_model = tree.DecisionTreeClassifier()
    elif class_type == 'LR':
        class_model = LogisticRegression(max_iter=200000)

    X_feature, Y_feature = train_feature.iloc[train_idx], test_feature.iloc[train_idx]
    X_feature, Y_feature = X_feature.T, Y_feature.T

    trained_model = class_model.fit(X_feature, train_label)

    return trained_model

# 进行K折交叉验证

def valid(m, class_type):
    epoch = 100

    # load features
    train_feature = pd.read_csv('path_to_feature.csv', header=None)
    test_feature = pd.read_csv('path_to_feature.csv', header=None)

    train_label = train_feature.iloc[:, 0]
    test_label = test_feature.iloc[:, 0]
    # print(test_label)
    class_num = int(np.max(test_label) + 1)


    train_feature = train_feature.drop(train_feature.columns[:1], axis=1)
    test_feature = test_feature.drop(test_feature.columns[:1], axis=1)
    random_seed_list = np.random.randint(0, 10000, size=epoch)


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

        # Create BCV partition set
        Mmax = 15
        n_feature = train_feature.shape[0]

        cross_validation_config = {"name": "MX2BCV", "m": Mmax,
                                   "n_size": n_feature}  # in this task, n_size is n_feature
        partitions = create_partitions_with_config(cross_validation_config)

        model_predict = None

        parallel = Parallel(n_jobs=-1)

        for train_idx, test_idx in partitions[:2 * m]:

            if class_type == 'MLP':
                class_model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=50, max_iter=200000)
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

    sum_confusion = sum_confusion / epoch

    test_acc = np.trace(sum_confusion[:, -class_num:]) / np.sum(sum_confusion[:, -class_num:])
    print('%s 2x%s test accuracy is %f:' % (class_type, m, test_acc))


    confusion_index = np.array([i + 1 for i in range(class_num)] * (2 * m + 1))

    pd_confusion = pd.DataFrame(sum_confusion)
    pd_confusion.columns = confusion_index
    pd_confusion.to_csv('bcv_confusion_matrix/model_nane/%s_2x%s.csv' % (class_type, m), header=True)


if __name__ == "__main__":
    class_type_list = ['DT', 'LR','MLP'
                       ]

    # Create process pool
    pool = multiprocessing.Pool(processes=4)
    start_time = time.time()
    # Submit tasks to the process pool
    m = 15

    for class_type in class_type_list:
        pool.apply_async(valid, args=(m, class_type))

    # Close the process pool and no longer accept new tasks
    pool.close()

    # Waiting for all tasks to be completed
    pool.join()
    print('running time: %s' % (time.time() - start_time))

