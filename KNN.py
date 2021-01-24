import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold

def dataframe_to_numpy(df):
    df['diagnosis'] = df['diagnosis'].replace(['M','B'],[0.,1.])
    return df.to_numpy()

def get_data(path="train.csv"):
    data =  pd.read_csv(path)
    return dataframe_to_numpy(data)

def compute_h(data):
    num_samples, num_features = data.shape
    num_features -= 1

    healthy = np.sum(data[:, 0])
    sick = len(data) - healthy
    total = len(data)
    if total == 0 :
        return 0

    h_healthy = healthy / total
    h_sick = sick / total

    healthy_part = h_healthy * np.log2(h_healthy) if h_healthy != 0 else 0
    sick_part = h_sick * np.log2(h_sick) if h_sick != 0 else 0

    h = -(healthy_part + sick_part)

    return h

def compute_IG_for_feature_threhold(data,threshold,feature):

    total = len(data)

    h_self = compute_h(data)

    under_threshold = data[:,feature] <= threshold
    above_threshold = data[:,feature] > threshold

    samples_under = data[under_threshold]
    samples_above = data[above_threshold]

    total_under = len(samples_under)
    h_under = compute_h(samples_under)

    total_above = len(samples_above)
    h_above = compute_h(samples_above)

    ig = h_self - ((total_under/total) *h_under + (total_above/total)*h_above)

    return ig

def find_best_feature_threshold_per(data,features_set):

    num_samples, num_features = data.shape
    num_features -= 1   # num_features shouldn't include index 0 where the label is

    best_feature = list(features_set)[0]
    best_threshold = 0
    best_ig = 0

    #todo I think it needs to move through f's we didn't move through
    # for f_index in range(1,num_features):
    for f_index in features_set:
        data = data[data[:, f_index].argsort()]

        for s_index in range(num_samples - 1):

            threshold = (data[s_index,f_index] + data[s_index + 1,f_index])/2

            ig = compute_IG_for_feature_threhold(data,threshold , f_index)

            if best_ig <= ig:
                best_ig = ig
                best_threshold = threshold
                best_feature = f_index

    return (best_ig, best_feature, best_threshold)


class Node:

    def __init__(self, data, features, parent=None, default_value=0, M=-1):

        # todo remove the 3 lines below

        self.average_sample = np.sum(data,axis=0)/len(data)
        self.m = M
        self.trues = -1
        if len(data) != 0:
            self.trues = [np.sum(data[:,0])/len(data), len(data), len(features)]

        self.parent = parent
        self.features = features
        self.children = []
        self.h = 0
        self.threshold = None
        self.feature_index = None

        if len(data) == 0:
            self.leaf = True
            self.default_value = default_value

            if len(data) > 0:
                self.h = compute_h(data)

            return

        if len(self.features) == 0:

            self.leaf = True
            self.default_value = 1 if np.sum(data[:, 0]) >= (len(data) / 2) else 0

            if len(data) > 0:
                self.h = compute_h(data)

            return

        healthy_leaf = np.sum(data[:,0]) == len(data)
        sick_leaf = np.sum(data[:,0]) == 0
        self.leaf = (healthy_leaf or sick_leaf)

        self.default_value = 1 if np.sum(data[:, 0]) >= (len(data)/2) else 0
        # self.default_value = 1 if np.sum(data[:, 0])*0.1 >= (len(data) - np.sum(data[:, 0])) else 0

        self.h = compute_h(data)


        if not self.split_condition(data):
            self.leaf = True
        else:
            if not self.leaf :
                self.threshold, self.feature_index = self.split(data, self.default_value)


    def split(self, data, default_value):

        if self.leaf:
            return None, None

        ig, feature, threshold = find_best_feature_threshold_per(data, self.features)
        self.features.remove(feature)

        split1 = data[data[:, feature] < threshold]
        split2 = data[data[:, feature] >= threshold]

        self.children = [
            Node(split1, self.features, self, default_value, M=self.m),
            Node(split2, self.features, self, default_value, M=self.m)
        ]
        # self.features.append(feature)
        return threshold, feature

    def predict(self, data): #the Node data structure is initialize with the feature set, and arguments it according to it's progression.
        if self.leaf:
            return self.default_value

        if len(data.shape) == 1:
            if data[self.feature_index] <= self.threshold:
                pred = self.children[0].predict(data)
            else:
                pred = self.children[1].predict(data)
            return pred

        predictions = []
        if len(data.shape) > 1:
            for i in range(data.shape[0]):
                d = data[i]
                thresh = d[self.feature_index] <= self.threshold
                if thresh:
                    pred = self.children[0].predict(d)
                else:
                    pred = self.children[1].predict(d)
                predictions.append(pred)

        return predictions

    def split_condition(self, data):
        if len(data) <= self.m and self.m != -1:
            return False
        return True

    def traverse(self, depth=0):
        print(self, "\n depth: ", depth)
        print()
        for c in self.children:
            c.traverse(depth+1)

    def __str__(self):
        return 'feature: {}\n threshold {}\n leaf {}\n trues {}'.format(self.feature_index, self.threshold, self.leaf, self.trues)



def get_k_fold_validation(data_train, n_splits=5):
    index_array = [sp for sp in KFold(n_splits=n_splits, random_state=203439989, shuffle=True).split(data_train)]
    validation_folds = [np.concatenate([[data_train[i]] for i in index_array[j][1]]) for j in range(len(index_array))]
    train_folds = [np.concatenate([[data_train[i]] for i in index_array[j][0]]) for j in range(len(index_array))]
    return train_folds, validation_folds


def get_id3_tree_from(data_train , M=-1):
    num_samples, num_features = data_train.shape
    num_features -= 1
    set_features = list(range(1, num_features))
    if M==-1:
        tree = Node(data_train, set_features)
    else:
        # print('hi')
        tree = Node(data_train, set_features,M=M)
    return tree


def test_id3_q1(data_train, data_test):
    # data_train = get_data("train.csv")
    # train predictions
    tree = get_id3_tree_from(data_train)

    # data_train = get_data("train.csv")
    # labels = data_train[:,0]
    # predictions = np.array(tree.predict(data_train))
    # print("accuracy on train set", np.sum(labels == predictions)/len(predictions))

    # test_predictions
    # data_test = get_data("test.csv")
    labels = data_test[:, 0]
    predictions = np.array(tree.predict(data_test))
    print("accuracy on test set", np.sum(labels == predictions) / len(predictions))
    print(np.sum(labels == predictions), " predictions were correct, out of: ", len(predictions))

# M is saved as a field in Node data structure.
# We use function: get_id3_tree_from(data_train,M) to get the tree
#
def experiment(data_train):
    # data_train = get_data("train.csv")
    # train predictions
    train_folds, validation_folds = get_k_fold_validation(data_train)

    m_array = [1, 3, 5, 10, 13]
    results = []

    for j in range(5):
        accuracy_results = []
        for i in range(5):
            tree = get_id3_tree_from(train_folds[i],M= m_array[j])
            labels = validation_folds[i][:, 0]
            predictions = np.array(tree.predict(validation_folds[i]))
            accuracy_results.append(np.sum(labels == predictions) / len(predictions))

        print('\ni: ', j, ' m is: ', m_array[j])
        print("Accuracy on test is:", np.average(accuracy_results))
        results.append(np.average(accuracy_results))
        # print(np.sum(labels == predictions), " predictions were correct, out of: ", len(predictions))
    plt.plot(m_array,results)
    plt.show()
    return


def test_id3_q3(data_train, data_test):
    # data_train = get_data("train.csv")
    # train predictions
    tree = get_id3_tree_from(data_train,M=10)

    # data_train = get_data("train.csv")
    # labels = data_train[:,0]
    # predictions = np.array(tree.predict(data_train))
    # print("accuracy on train set", np.sum(labels == predictions)/len(predictions))

    # test_predictions
    # data_test = get_data("test.csv")
    labels = data_test[:, 0]
    predictions = np.array(tree.predict(data_test))
    print("Accuracy on test set", np.sum(labels == predictions) / len(predictions))
    print(np.sum(labels == predictions), " predictions were correct, out of: ", len(predictions))

def test_id3_q4(data_train, data_test):
    # data_train = get_data("train.csv")
    # train predictions
    tree = get_id3_tree_from(data_train,M=10)

    # data_train = get_data("train.csv")
    # labels = data_train[:,0]
    # predictions = np.array(tree.predict(data_train))
    # print("accuracy on train set", np.sum(labels == predictions)/len(predictions))

    # test_predictions
    # data_test = get_data("test.csv")
    labels = data_test[:, 0]
    predictions = np.array(tree.predict(data_test))

    n = len(predictions)

    fp = 0
    fn = 0

    for i in range(n):
        if (labels[i] == 1 and predictions[i] == 0):
            fn +=1
        elif labels[i] == 0 and predictions[i] == 1:
            fp += 1
    loss = (0.1*fp + fn)/n
    print("The loss is: ",loss)
    # print(np.sum(labels == predictions), " predictions were correct, out of: ", len(predictions))

def rand_samples_choice(data, n_trees, p):

    training_data_list = []

    for i in range(n_trees):
        n = len(data)
        indices = np.random.choice(n, int(n*p), replace=False)
        training_data_list.append(data[indices])

    return training_data_list

class KNN_forest:

    def __init__(self, data_train, n_trees=5, k=3, p=0.5):

        # train_folds, validation_folds = get_k_fold_validation(data_train, n_trees)
        fold = rand_samples_choice(data_train, n_trees, p)

        # print("fold ",fold[0].shape, type(fold), len(fold))

        self.k = k

        self.trees = []
        self.avg_sample = []
        for i in range(n_trees):

            num_samples, num_features = fold[i].shape
            set_features = list(range(1, num_features - 1))

            tree = Node(fold[i], set_features)
            self.trees.append(tree)
            self.avg_sample.append(np.sum(fold[i], axis=0)/len(fold[i]))

        self.avg_sample = np.array(self.avg_sample)

    def predict(self, data_p):

        results = []
        # print(data_p.shape)

        if len(data_p.shape) == 1:
            data_p = np.expand_dims(data_p, 0)

        for data in data_p:

            # find out which trees to use

            distance = np.sum((self.avg_sample[:, 1:] - data[1:])**2, axis=1) # no need to compute root we only want the K best
            best_distances = np.argsort(distance)[:self.k]

            # use k closest trees to compute

            k_tree_results = []
            for index in best_distances:
                tree_result = self.trees[index].predict(data)
                k_tree_results.append(tree_result)

            k_tree_results = np.array(k_tree_results)
            result = 1 if np.sum(k_tree_results) > self.k - np.sum(tree_result) else 0
            results.append(result)

        return results

        # for tree in self.trees:
        #     print(tree.predict(data))



def find_best_knn_parameters(data_train, data_test):


    n = len(data_train[:, 0])
    # print(n)

    prediction_accuracy_max = 0
    max_p = 0
    max_n_trees = 1
    max_k = 1
    p0 = 0.5
    while p0<=0.7:
        N = int(p0*n)
        for i in range(2,N+1,2):
            # print('i: ',i)
            for n_trees in range(1,N+1):
                # print('n_trees: ', n_trees)
                for k in range(1,n_trees+1):
                    # print('k: ', k)
                    forest = KNN_forest(data_train, n_trees=n_trees, k=k, p=p0)
                    results = np.array(forest.predict(data_test))
                    labels = np.array([int(k) for k in data_test[:, 0]])
                    prediction_acuracy = np.sum(labels == results) / len(results)
                    # print("predictions: ", prediction_acuracy)
                    if prediction_accuracy_max <= prediction_acuracy:
                        print('------------ In Max! ------------')
                        prediction_accuracy_max = prediction_acuracy
                        max_p = p0
                        max_k = k
                        max_n_trees = n_trees
                        print('this is max_p: ', max_p, '\nthis is max_k: ', max_k, '\nthis is max_n_trees: ',
                              max_n_trees,
                              '\nthis is prediction_accuracy_max: ', prediction_accuracy_max)
        p0 += 0.05
    return max_p, max_k, max_n_trees, prediction_accuracy_max




data_train = get_data("train.csv")
data_test = get_data("test.csv")

max_p, max_k, max_n_trees, prediction_accuracy_max = find_best_knn_parameters(data_train, data_test)

print('---- Done! ----')
print('this is max_p: ',max_p,'\nthis is max_k: ',max_k,'\nthis is max_n_trees: ',max_n_trees,
      '\nthis is prediction_accuracy_max: ',prediction_accuracy_max)


# test_id3_q1(data_train,data_test)
# experiment(data_train)
# test_id3_q4(data_train,data_test)

# index_array= [sp for sp in KFold(n_splits=5, random_state=203439989, shuffle=True).split(data_train)]
# validation_folds = [np.concatenate([[data_train[i]] for i in index_array[j][1]]) for j in range(len(index_array))]
# train_folds = [np.concatenate([[data_train[i]] for i in index_array[j][0]]) for j in range(len(index_array))]
# print(train_folds[-1].shape)



# forest = KNN_forest(data_train,n_trees=150, k=3, p=0.7)
#
# results = np.array(forest.predict(data_test))
# labels = np.array([int(k) for k in data_test[:,0]])
# prediction_accuracy = np.sum(labels == results)/len(results)


# print("predictions: ", prediction_accuracy)

# a = np.arange(10*31).reshape(10,31)
# b = np.random.choice(10,3,replace=False)
# print(a,b)
# print(a.shape)
# print(a[b].shape)
# print(a[b])