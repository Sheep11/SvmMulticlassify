import numpy as np


def load_training_data(filepath):
    i = 0
    data = []
    data1 = []
    data2 = []
    data3 = []

    with open(filepath) as f:
        for line in f.readlines():  # get data by lines
            if line == "\n":    # empty line
                break

            i += 1
            feature = line.split(',')
            feature.pop()   # abandon last element, because it is label

            if 1 <= i <= 30:
                data1.append(list(map(float, feature)))  # append to data

            if 51 <= i <= 80:
                data2.append(list(map(float, feature)))  # append to data

            if 101 <= i <= 130:
                data3.append(list(map(float, feature)))  # append to data

    data.append(data1)
    data.append(data2)
    data.append(data3)

    return data
    # return np.array(data)


def load_test_data(path):
    data = []
    label = []
    i = 0
    with open(path) as f:
        for line in f.readlines():  # get data by lines
            if line == "\n":    # empty line
                break
            i += 1

            if 30 < i <= 50 or 80 < i <= 100 or 130 < i <= 150:
                if 30 < i <= 50:
                    label.append(0)
                elif 80 < i <= 100:
                    label.append(1)
                elif 130 < i <= 150:
                    label.append(2)

                feature = line.split(',')
                feature.pop()   # abandon last element, because it is label
                data.append(list(map(float, feature)))  # append to data

    return np.array(data), label

# if __name__ == "__main__":
#     data_set = load_data('data/iris.data')
#     print(data_set)
#     print(data_set[2][39])
