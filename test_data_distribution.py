from scaled.dataset.sfc_dataset import SFCDiffusionDataset


train_dataset = SFCDiffusionDataset(
        data_dir="data/SFC/SFC_data_csv",
        data_list=[i for i in range(5, 3500)])
val_dataset  = SFCDiffusionDataset(
    data_dir="data/SFC/SFC_data_csv",
    data_list=[i for i in range(3500,3990)])


data_0,data_1 = train_dataset[0]

data0 = data_0[0]

data1 = data_0[1]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 创建图形
plt.figure(figsize=(8, 5))
# 直方图 + KDE（核密度估计）
sns.histplot(data0, bins=30, kde=True, color='blue', edgecolor='black', alpha=0.7)

# 添加标题和标签
plt.title("1D Data Distribution", fontsize=14)
plt.xlabel("Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# 显示图像
plt.savefig('result_v1.png')
plt.close()

# 创建图形
plt.figure(figsize=(8, 5))
# 直方图 + KDE（核密度估计）
sns.histplot(data1, bins=30, kde=True, color='blue', edgecolor='black', alpha=0.7)

# 添加标题和标签
plt.title("1D Data Distribution", fontsize=14)
plt.xlabel("Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# 显示图像
plt.savefig('result_v2.png')