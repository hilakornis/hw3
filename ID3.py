import numpy as np
import pandas as pd

def dataframe_to_numpy(df):
    df['diagnosis'] = df['diagnosis'].replace(['M','B'],[0.,1.])
    return df.to_numpy()

def get_train(path="train.csv"):
    data =  pd.read_csv(r'train.csv')
    return dataframe_to_numpy(data)

# def compute_h(total, healthy, sick):
#
#     h_healthy = healthy / total
#     h_sick = sick / total
#
#     h = -(h_healthy * np.log2(h_healthy) + h_sick * np.log2(h_sick))
#
#     return h

def compute_h(data):
    num_samples, num_features = data.shape
    num_features -= 1

    healthy = np.sum(data[:, 0])
    sick = len(data) - healthy
    total = len(data)

    h_healthy = healthy / total
    h_sick = sick / total

    h = -(h_healthy * np.log2(h_healthy) + h_sick * np.log2(h_sick))

    return h

def compute_IG_for_feature_threhold(data,threshold,feature):
    num_samples, num_features = data.shape
    num_features -= 1

    healthy_self = np.sum(data[:,0])
    sick_self = len(data) - healthy_self
    total = len(data)

    h_self = compute_h(data) #compute_h(total, healthy_self, sick_self)

    under_threshold = data[:,feature] < threshold
    above_threshold = data[:,feature] > threshold

    samples_under = data[under_threshold]
    samples_above = data[above_threshold]

    # healthy_under = np.sum(samples_under[:,0])
    # sick_under = len(samples_under) - healthy_under
    #
    total_under = len(samples_under)
    h_under = compute_h(samples_under)#total_under, healthy_under, sick_under)

    # healthy_above = np.sum(samples_above[:, 0])
    # sick_above = len(samples_above) - healthy_above
    total_above = len(samples_above)


    h_above = compute_h(samples_above) #total_above, healthy_above, sick_above)

    ig = h_self - ((total_under/total) *h_under + (total_above/total)*h_above)

    return ig

class Node:

    def __init__(self, data, parent=None, default_value=0):
        self.parent = parent

        if len(data) == 0:
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

    # def compute_h(self, data):
    #     num_samples, num_features = data.shape
    #     num_features -= 1
    #
    #     healthy = np.sum(data[:, 0])
    #     sick = len(data) - healthy
    #     total = len(data)
    #
    #     h_healthy = healthy / total
    #     h_sick = sick / total
    #
    #     h = -(h_healthy * np.log2(h_healthy) + h_sick * np.log2(h_sick))
    #
    #     return h


data = get_train()
print(data, data.shape)
print(compute_IG_for_feature_threhold(data,11,1))

print(Node(data))
# print(Node.compute_h(Node, data))