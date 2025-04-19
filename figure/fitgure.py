import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
# 模拟数据
K = np.arange(3, 10, 2)
bar_width = 0.25

# 每张子图的指标数据（你可以根据实际替换）
metrics_data = {
    "Recall@K": {
        "MPGCF-S-BERT": [0.37793 , 0.42827 , 0.47462 , 0.50705],
        "MPGCF-ACC":     [ 0.11653 , 0.15008 , 0.16603 , 0.20743],
        "MPGCF-NACC":    [0.39323 , 0.48303 , 0.53661 , 0.55161],
        "MPGCF-SEM":     [0.38514 , 0.48262 , 0.52345 , 0.56345],
        "MPGCF-STR":     [0.35503 , 0.47582 , 0.51901 , 0.53026],
        "MPGCF":         [0.40388 , 0.49908 , 0.54641 , 0.57837]
    },
    "NDCG@K": {
        "MPGCF-S-BERT": [0.32113 , 0.33903 , 0.35614 ,0.36582],
        "MPGCF-ACC":     [0.09721 , 0.11393 , 0.11662 , 0.13737],
        "MPGCF-NACC":    [0.34048 , 0.37885 ,0.40140 , 0.41328 ],
        "MPGCF-SEM":     [0.34374 , 0.36960 , 0.39527 , 0.41668],
        "MPGCF-STR":     [0.33118 , 0.35177 , 0.37321 , 0.39135],
        "MPGCF":         [0.35125 , 0.40283 , 0.41387 , 0.43487]
    },
    "Coverage@K": {
        "MPGCF-S-BERT": [0.37762 , 0.48117 , 0.56276 , 0.61506],
        "MPGCF-ACC":     [0.72594 , 0.86925 , 0.91213 , 0.97071],
        "MPGCF-NACC":    [0.43096 ,0.56799  , 0.65841 , 0.75418],
        "MPGCF-SEM":     [0.59167 , 0.66485 , 0.70364 , 0.77510],
        "MPGCF-STR":     [0.56276 , 0.59698 , 0.62238 , 0.71778],
        "MPGCF":         [0.63494 , 0.69665 , 0.76569 , 0.83996]
    },
    "Tail@K": {
        "MPGCF-S-BERT": [0.09433 , 0.11952 , 0.12782 , 0.13260],
        "MPGCF-ACC":     [0.69237 , 0.72100 , 0.72041 , 0.71670],
        "MPGCF-NACC":    [0.20590 , 0.12214 , 0.16048 , 0.13134],
        "MPGCF-SEM":     [0.22114  , 0.14387  , 0.17345  , 0.21245],
        "MPGCF-STR":     [0.18628 , 0.15564 , 0.19739 , 0.21463],
        "MPGCF":         [ 0.24920 , 0.19901 , 0.23350 , 0.24048]
    }
}

# 设置样式
colors = ['#fdbf6f', '#ffff99', '#a6d854', '#80b1d3', '#d580ff', '#b15928']
hatches = ['', 'xxx', '///', '...', '|||', '---']
model_names = list(metrics_data["Tail@K"].keys())

# 创建大图和子图
fig, axs = plt.subplots(2, 2, figsize=(14, 11))

plt.subplots_adjust(
    wspace=0.3,  # 水平间距
    hspace=0.3,  # 垂直间距
    left=0.07,   # 左边距
    right=0.90,  # 右边距
    top=0.98,    # 上边距
    bottom=0.09  # 下边距
)

axs = axs.flatten()

# 每个子图绘图
for idx, (metric_name, data) in enumerate(metrics_data.items()):
    ax = axs[idx]
    for i, model in enumerate(model_names):
        ax.bar(K + i * bar_width - bar_width * 2.5, data[model],
               width=bar_width,
               label=model,
               color=colors[i],
               hatch=hatches[i],
               edgecolor='black')

    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xticks(K)
    ax.set_xticklabels(K)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # 每张子图添加图例
    ax.legend(
        loc='upper left',
        fontsize=10,
        ncol=2,
        frameon=True
    )
    if idx == 0:  # Recall
        ax.set_ylim(0, 0.7)
    elif idx == 1:  # NDCG
        ax.set_ylim(0, 0.55)
    elif idx == 2:  # Coverage
        ax.set_ylim(0, 1.1)
    elif idx == 3:  # Tail
        ax.set_ylim(0, 0.9)


    if idx==0:
        metric_name = "(a) Recall"
    elif idx==1:
        metric_name = "(b) NDCG"
    elif idx==2:
        metric_name = "(c) Coverage"
    elif idx==3:
        metric_name = "(d) Tail"
    # 设置标题的位置（下方）
    ax.text(0.5, -0.2, metric_name, fontsize=18, ha='center', transform=ax.transAxes)


plt.savefig('output_figure.png', dpi=600, bbox_inches='tight')  # PNG格式，高清
# plt.savefig('output_figure.pdf', bbox_inches='tight')         # PDF格式，适合论文
plt.show()
#
# # plt.tight_layout()
# plt.show()
