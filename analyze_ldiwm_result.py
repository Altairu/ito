import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# 日本語フォント
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
# データ読み込み
df = pd.read_csv("実験4結果/kadai4_ldiwm_history.csv", index_col=0)

# 各試行の最終評価値と最小評価値
result_summary = {
    "試行番号": [],
    "最終評価値": [],
    "最小評価値": [],
    "最小値を達成した世代": []
}

for col in df.columns:
    result_summary["試行番号"].append(col)
    result_summary["最終評価値"].append(df[col].iloc[-1])
    result_summary["最小評価値"].append(df[col].min())
    result_summary["最小値を達成した世代"].append(df[col].idxmin())

summary_df = pd.DataFrame(result_summary)

# 保存先
output_dir = "実験4結果"
summary_df.to_csv(os.path.join(output_dir, "kadai4_ldiwm_解析結果.csv"), index=False)

# 表示
print(summary_df)

# オプション：最小評価値のヒストグラム
plt.figure(figsize=(8,5))
plt.hist(summary_df["最小評価値"], bins=10, edgecolor='black')
plt.xlabel("最小評価値")
plt.ylabel("頻度")
plt.title("LDIWM付きPSOの最小評価値分布")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kadai4_ldiwm_最小評価値ヒストグラム.png"))
plt.show()
