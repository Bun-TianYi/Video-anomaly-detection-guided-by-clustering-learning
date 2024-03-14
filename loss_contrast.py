import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = "loss_record"
# mix = np.load(os.path.join(path, "mix_loss_data.npy"))
mix = np.load(os.path.join(path, "mix_Shan.npy"))
mix = mix[0:60]
# no_conv = np.load(os.path.join(path, "no_conv_loss_data222.npy"))
# no_attn = np.load(os.path.join(path, "no_attn_loss_data.npy"))
no_attn = np.load(os.path.join(path, "no_attn_Shan.npy"))
no_attn = no_attn[0:60]
no_conv = np.load(os.path.join(path, "no_conv_Shan.npy"))
no_conv = no_conv[0:60]

plt.figure(figsize=(6, 6))
# plt.plot(no_conv, label='no_conv')
plt.plot(no_attn, label='no_attn')
plt.plot(mix, label='mix')
plt.plot(no_conv, label='no_conv')

plt.legend()
plt.grid()
plt.show()
