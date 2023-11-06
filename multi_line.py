import numpy as np


def calculate_cost_matrix(x, y):
    cumsum_x = [0] + [sum(x[:i + 1]) for i in range(n)]
    cumsum_y = [0] + [sum(y[:i + 1]) for i in range(n)]
    cumsum_x_squared = [0] * (n + 1)
    cumsum_y_squared = [0] * (n + 1)
    cumsum_xy = [0] * (n + 1)

    for i in range(1, n + 1):
        cumsum_x_squared[i] = cumsum_x_squared[i - 1] + x[i - 1] ** 2
        cumsum_y_squared[i] = cumsum_y_squared[i - 1] + y[i - 1] ** 2
        cumsum_xy[i] = cumsum_xy[i - 1] + x[i - 1] * y[i - 1]


    err = np.zeros((n, n))
    for j in range(n):
      for i in range(j + 1):
            interval = j - i + 1

            x_sum = cumsum_x[j + 1] - cumsum_x[i]
            y_sum = cumsum_y[j + 1] - cumsum_y[i]
            xy_sum = cumsum_xy[j + 1] - cumsum_xy[i]
            xsqr_sum = cumsum_x_squared[j + 1] - cumsum_x_squared[i]
            ysqr_sum = cumsum_y_squared[j + 1] - cumsum_y_squared[i]

            num = (interval * xy_sum) - (x_sum * y_sum)
            if num == 0:
              m = 0.0
            else:
              denom = (interval * xsqr_sum) - (x_sum * x_sum)
              m = float('inf') if denom == 0 else num/ denom

            c = (y_sum - (m * x_sum)) / interval

            err[i, j] = (m * m * xsqr_sum) + ysqr_sum + (2 * m * c * x_sum) - (2 * m * xy_sum) + (interval * c * c) - (2 * c * y_sum)

    return err


# def calculate_error(line, points):
#     a, b = line
#     if a == 0 and b == 0: return 0
#     xi, yi = np.array(points).T
#     predicted_y = a * xi + b
#     errors = (yi - predicted_y) ** 2
#     error = np.sum(errors)

#     return error


def multi_line_fitting(X, Y, C):
    n = len(X)



    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    cost_function = calculate_cost_matrix(X, Y)

    partitions = [0] * (n + 1)


    for j in range(1, n+1):
        k = 0
        for i in range(1, j + 1):
            if dp[j] > dp[i - 1] + cost_function[i-1][j-1] + C:
              dp[j] = dp[i - 1] + cost_function[i-1][j-1] + C
              k = i
        partitions[j] = k



    segments = []
    i = n
    j = partitions[n]
    while i > 0:
        segments.append((i, j))
        i = j - 1
        j = partitions[i]

    cuts = []
    for i, j in reversed(segments):
        cuts.append((i - 1))


    return dp[n], cuts


import pickle
file_path = 'test_set_large_instances'
examples_of_instances = pickle.load(open(file_path, 'rb'))

k_list =[]

last_points_list = []

OPT_list = []

test = len(examples_of_instances['n_list'])

for i in range(test):
  n = examples_of_instances['n_list'][i]
  C =  examples_of_instances['C_list'][i]
  X =  examples_of_instances['x_list'][i]
  Y = examples_of_instances['y_list'][i]

  min_cost, cuts = multi_line_fitting(X, Y, C)

  k_list.append(len(cuts))

  last_points_list.append(cuts)

  OPT_list.append(min_cost)

output_solutions = {
    "k_list": k_list,
    "last_points_list": last_points_list,
    "OPT_list": OPT_list
}
file_path = 'large_solutions' 
# Open the file in binary write mode ("wb") and dump the dictionary into it
with open(file_path, 'wb') as file:
    pickle.dump(output_solutions, file)