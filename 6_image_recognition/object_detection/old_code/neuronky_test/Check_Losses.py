# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import pandas as pd

# %%
def load_loss_data(path):
    with open(path + "/metric_logger_iter.pickle", 'rb') as f:
        dict = pickle.load(f)
    return(dict)

def plot_loss(full_dick):
    legend_name = full_dick["full_path"].split("/")[3].split("_")[2]
    print(legend_name)

    # if (legend_name == "09") or (legend_name == "06"):
    #     return None

    # if legend_name != "1200":
    #     return None

    plot_data = [x["logs"]["loss"] for x in full_dick["dicts"]]

    plt.plot(np.arange(len(plot_data)), plot_data, label=legend_name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of data shrinked")
    plt.legend()
    # plt.xticks(np.arange(0, 20, step=1))
    # plt.yticks(np.arange(0, 0.35, 0.05))


def plot_loss_masks(full_dick):
    legend_name = full_dick["full_path"].split("/")[3].split("_")[2]

    # if (legend_name == "09") or (legend_name == "06"):
    #     return None

    # if legend_name != "1200":
    #     return None

    plot_data = [x["logs"]["loss_mask"] for x in full_dick["dicts"]]

    plt.plot(np.arange(len(plot_data)), plot_data, label=legend_name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of data shrinked")
    plt.legend()
    # plt.xticks(np.arange(0, 20, step=1))

#%%

path_to_folders = os.listdir("./tmp/saved_metrics/")
full_path = pd.DataFrame(path_to_folders, columns = ["full_path"]).apply(lambda y: "./tmp/saved_metrics/" + y)
full_path["dicts"] = full_path.applymap(lambda g: load_loss_data(g))

# %%
full_path.apply(lambda g: plot_loss(g), axis = 1)
plt.show()

full_path.apply(lambda g: plot_loss_masks(g), axis = 1)
plt.show()

# %%
path_to_metrics = "./tmp/saved_metrics/saved_pixels_feature_extra/metric_logger_iter.pickle"
path_to_metrics_09 = "./tmp/saved_metrics/saved_models_08/metric_logger_iter.pickle"

with open(path_to_metrics, 'rb') as f:
    dict = pickle.load(f)

with open(path_to_metrics_09, 'rb') as f:
    dict_09 = pickle.load(f)

plot_data = [x["logs"]["loss"] for x in dict]
plot_data_09 = [x["logs"]["loss"] for x in dict_09]

plt.plot(np.arange(20), plot_data, color = "r", label = "05")
plt.plot(np.arange(20), plot_data_09, color = "g", label = "09")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss of data shrinked by 0.5 or 0.9")
plt.legend()
plt.xticks(np.arange(0, 20, step=1))
plt.show()

