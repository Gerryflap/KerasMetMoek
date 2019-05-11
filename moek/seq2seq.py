import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Example: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
# Load dataset
dataset = pd.read_csv("114-time_cleaned_data.csv")
# Take only data from this week (4 15 minutes frame in hour: 4*24*7=672) (One month would be 2688)
dataset = dataset[:672]
print(len(dataset))

# View first 5 entries
# print(dataset.head(500))


def print_difference_between_missing_data(filename):
    #  See https://stackabuse.com/how-to-format-dates-in-python/ for good explanation on datetime formatting
    fmt = "%Y-%m-%d %H:%M:%S"
    dataset = pd.read_csv(filename)
    for index, row in dataset.iterrows():
        if index == 0:
            prev_time = row["local_15min"]
            continue
        diff = datetime.strptime(row["local_15min"], fmt) - datetime.strptime(prev_time, fmt)
        if diff.total_seconds() == 900:
            prev_time = row["local_15min"]
            continue
        else:
            print(prev_time, "vs", row["local_15min"])
            print(diff)
            prev_time = row["local_15min"]


def calculate_percentage_missing(filename):
    #  See https://stackabuse.com/how-to-format-dates-in-python/ for good explanation on datetime formatting
    correct = 0
    incorrect = 0
    fmt = "%Y-%m-%d %H:%M:%S"
    dataset = pd.read_csv(filename)
    for index, row in dataset.iterrows():
        if index == 0:
            prev_time = row["local_15min"]
            continue
        diff = datetime.strptime(row["local_15min"], fmt) - datetime.strptime(prev_time, fmt)
        if diff.total_seconds() == 900:
            correct += 1
            prev_time = row["local_15min"]
            continue
        else:
            incorrect += diff.total_seconds() / 900
            prev_time = row["local_15min"]
    return incorrect / (correct + incorrect)


def check_weather_data(filename):
    #  See https://stackabuse.com/how-to-format-dates-in-python/ for good explanation on datetime formatting
    correct = 0
    incorrect = 0
    fmt = "%Y-%m-%d %H:%M:%S+%f"
    dataset = pd.read_csv(filename)
    for index, row in dataset.iterrows():
        if index == 0:
            prev_time = row["localhour"]
            continue
        diff = datetime.strptime(row["localhour"], fmt) - datetime.strptime(prev_time, fmt)
        if diff.total_seconds() == 3600:
            correct += 1
            prev_time = row["localhour"]
            continue
        else:
            print(prev_time, "vs", row["localhour"])
            print(diff)
            incorrect += diff.total_seconds() / 3600
            prev_time = row["localhour"]
    return incorrect / (correct + incorrect)


print(check_weather_data("weather2014.csv"))
# print(calculate_percentage_missing("114-time_cleaned_data.csv"))

# values = dataset.values
# # specify columns to plot
# groups = [2, 3, 8, 10, 15, 16, 20, 25, 29, 30, 37, 38, 39, 45, 47, 59]
# i = 1
# # plot each column
# plt.figure()
# for group in groups:
#     # plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])
#     plt.title(dataset.columns[group], y=0.5, loc='right')
#     i += 1
#     plt.show()


