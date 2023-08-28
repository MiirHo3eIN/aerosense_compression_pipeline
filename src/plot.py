import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# data preparation
df_models = pd.read_csv("../models.csv", usecols=['model_id', 'latent_channels', 'latent_seq_len', 'alpha'])
df_acc = pd.read_csv("../results/classification_accuracy_v1.csv", usecols=['model_id', 'rfc', 'ridge'])
df_pre = pd.read_csv("../results/precision_results_v1.csv", usecols=['model_id', 'rfc', 'ridge'])


models = ()
rid_acc = ()
rfc_acc = ()

for i in range(len(df_models)):
    dim_str = f"{df_models.iloc[i]['latent_channels']} x {df_models.iloc[i]['latent_seq_len']} \n\
a = {df_models.iloc[i]['alpha']}"
    models += (dim_str,)
    rid_acc += (df_acc.iloc[i]['ridge'], )
    rfc_acc += (df_acc.iloc[i]['rfc'], )

data = {
    'Ridge': rid_acc,
    'Random Forest': rfc_acc,
}

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0.5

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, fmt='{:.2}', padding=2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Acc')
ax.set_title('Accuracy of CNNs with different Latent Space Dimensions')
ax.set_xticks(x + width, models)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 0.5)

plt.show()

