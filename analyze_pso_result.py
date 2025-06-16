import pandas as pd
import os
import matplotlib.pyplot as plt

# 日本語フォント
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# ファイル読み込み
input_path = "./実験1結果/収束履歴表.csv"
df = pd.read_csv(input_path, index_col=0)

# 解析結果保存フォルダ
analysis_dir = "./実験1結果/解析結果"
os.makedirs(analysis_dir, exist_ok=True)

# 統計解析用データ
summary = {
    "関数名": [],
    "最終評価値": [],
    "最小評価値": [],
    "最小値を達成した世代": []
}

# 差分プロット用
plt.figure(figsize=(10, 6))

for col in df.columns:
    data = df[col]
    final_val = data.iloc[-1]
    min_val = data.min()
    min_gen = data.idxmin()

    summary["関数名"].append(col)
    summary["最終評価値"].append(final_val)
    summary["最小評価値"].append(min_val)
    summary["最小値を達成した世代"].append(min_gen)

    # 差分（収束安定性）可視化
    diff = data.diff().abs()
    plt.plot(diff, label=col)

# 統計結果をCSV保存
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(analysis_dir, "解析結果.csv"), index=False)

# 差分グラフ保存
plt.xlabel("世代")
plt.ylabel("前世代との絶対差")
plt.title("PSO収束の安定性（世代間差分）")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "差分グラフ.png"))
plt.show()

# コンソール出力（確認用）
print(summary_df)
