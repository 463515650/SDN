import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE



def t_sne(data, label):
    X = data
    y = label

    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # 自定义颜色
    colors = ListedColormap(plt.cm.tab10.colors)

    # 可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colors, edgecolor='k', alpha=0.5)
    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # 添加图例
    plt.legend(*scatter.legend_elements(), title="Classes")

    plt.show()



