import numpy as np
import pandas as pd

def dataframe_to_numpy(df):
    df['diagnosis'] = df['diagnosis'].replace(['M','B'],[0.,1.])
    return df.to_numpy()

def get_train(path="train.csv"):
    data =  pd.read_csv(r'train.csv')
    return dataframe_to_numpy(data)

def compute_h(data):
    num_samples, num_features = data.shape
    num_features -= 1

    healthy = np.sum(data[:, 0])
    sick = len(data) - healthy
    total = len(data)
    if total == 0 :
        return 0
    # assert total > 0


    # print(total)
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

def find_best_feature_threshold_per(data):
    num_samples, num_features = data.shape
    num_features -= 1   # num_features shouldn't include index 0 where the label is

    best_feature = 0
    best_threshold = 0

    best_ig = 0

    for f_index in range(1,num_features):
        data = data[data[:, f_index].argsort()]

        for s_index in range(num_samples - 1):

            # print()
            # print(s_index, f_index)
            # print(data[s_index - 1, f_index])
            # print(data[s_index, f_index])
            # print(data[s_index + 1, f_index])

            threshold = (data[s_index,f_index] + data[s_index + 1,f_index])/2
            # print('this is threshold ',threshold)
            ig = compute_IG_for_feature_threhold(data,threshold , f_index)

            if best_ig < ig:
                best_ig = ig
                best_threshold = threshold
                best_feature = f_index

    return (best_ig, best_feature, best_threshold)


class Node:

    def __init__(self, data, parent=None, default_value=0):

        self.parent = parent

        if len(data) == 0 or len(data[0,:]) == 1 :
            self.leaf = True
            self.default_value = default_value
            return

        healthy_leaf = np.sum(data[:,0]) == len(data)
        sick_leaf = np.sum(data[:,0]) == len(data)

        self.leaf = (healthy_leaf or sick_leaf)
        self.default_value = 1 if np.sum(data[:, 0]) >= (len(data)/2) else 0

        self.h = compute_h(data)

        self.feature_index = None
        self.threshold = None

        self.children = []

        print(self.h)




data = get_train()
print(data, data.shape)
print(compute_IG_for_feature_threhold(data,11,1))

print(Node(data))
# print(Node.compute_h(Node, data))

# print(data[data[:,0].argsort()])
print(find_best_feature_threshold_per(data))
