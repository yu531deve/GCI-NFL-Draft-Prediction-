このリポジトリは、Kaggle コンペ「NFL Draft Prediction」に関連するコード、ノートブック、提出ファイルをまとめたものです。

## /

## 📁 notebook 目次

# 00_baseline(0.80792).ipynb

ベースラインモデル。
・欠損値補完はすべて平均値
・特徴量に BMI を追加
・モデルはランダムフォレスト

# 01_0620_preprocessing.ipynb

主に EDA を行ったファイル
・欠損値を補完せずに削除した
・スコアは大幅に下がった

# 02_0621_notebook(0.79804).ipynb

ベースラインモデルに新たな特徴量を加えたモデル
・特徴量として Sprint_40yd_missing，Sprint_40yd_and_Bench_missing を追加
・AUC は伸びたが public の方は下がってしまった。

# 03_0624_notebook().ipynb

🔧 データ前処理・特徴量エンジニアリングまとめ
Id 列を削除（識別子のため）

目的変数 Drafted を X から分離して y に格納

Age 欠損に対して以下の 2 軸で処理：

Age_filled（-1 で補完）

Age_missing（欠損フラグ）

その他の数値列は中央値で補完

School, Player_Type, Position_Type, Position は Label Encoding

BMI を新規特徴量として導入（Weight / (Height^2)）

Player_Type に対して Target Encoding を適用

以下の特徴量を削除（重要度が低い・過学習の要因と判断）：

Age_missing, Player_Type, Position_Type, School

⚙️ モデル構築・評価
モデル：LightGBM（以下の過学習対策を適用）

木の深さ制限（max_depth=3, num_leaves=7）

学習率減少（learning_rate=0.05）と木数増加（n_estimators=500）

L1/L2 正則化（reg_alpha=2.0, reg_lambda=2.0）

サブサンプリング（subsample=0.8, colsample_bytree=0.8）

スプリット閾値追加（min_split_gain=0.1）

評価方法：KFold（5 分割） + AUC スコア + EarlyStopping（50 ラウンド）

📈 現在の評価結果（例）
平均 Train AUC：0.9166

平均 Validation AUC：0.8254

# 04

# 05

---

## 📁 プロジェクト構成

<details>
<summary>▼ クリックして展開</summary>

```
nfl-draft-prediction/
├── .venv/ # 仮想環境（Git 除外推奨）
├── data/ # Kaggle 公式データ（.gitignore で除外）
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
├── models/ # 保存済みモデル（.gitignore で除外）
│ └── .pkl など
├── notebooks/ # 分析・実験用ノートブック
│ ├── 00_baseline.ipynb
│ ├── 01_preprocessing.ipynb
│ └── catboost_info/ # CatBoost の学習ログ（自動生成）
│ ├── catboost_training.json
│ └── learn/
│ ├── events.out.tfevents
│ ├── learn_error.tsv
│ └── time_left.tsv
├── output/ # グラフなどの出力（任意・.gitignore 推奨）
│ └── .png 等
├── src/ # 再利用スクリプト
│ ├── features.py
│ ├── model.py
│ └── pycache/ # Python キャッシュ（Git 除外）
│ ├── features.cpython-.pyc
│ └── model.cpython-.pyc
├── submissions/ # 提出ファイルの保存場所
│ └── *.csv
├── .gitignore # 除外定義（data/, models/, pycache/ など）
├── README.md # 本ファイル
├── README.ipynb # Markdown 編集用の補助ノートブック（任意）
└── requirements.txt # 使用ライブラリ一覧
```

</details>

---

## 📓 ノートブック命名ルール

| ファイル名                | 内容                                   |
| ------------------------- | -------------------------------------- |
| `01_eda.ipynb`            | 初期の可視化と仮説立案                 |
| `02_preprocessing.ipynb`  | 欠損値補完、カテゴリ変数の処理など     |
| `03_model_baseline.ipynb` | ランダムフォレストなどによる初期モデル |
| `04_lgb_optuna.ipynb`     | LightGBM + Optuna によるチューニング   |
| `05_stack_ensemble.ipynb` | スタッキングやアンサンブルの実験       |

---

## 🗂 ファイルの分類方針

- `data/`, `models/`：頻繁に変化するローカルデータ（**.gitignore で除外**）
- `notebooks/`：ノートブックによる実験記録（**ステップ順に命名**）
- `src/`：再利用コード（`.py` に整理して `import`）
- `submissions/`, `output/`：提出ファイルやグラフなどの出力
- `README.md`, `requirements.txt`：プロジェクト構成情報と環境再現用

---

## 🚀 使用方法

1. 本リポジトリをクローンする：

   ```bash
   git clone https://github.com/yourname/nfl-draft-prediction.git
   cd nfl-draft-prediction
   ```

2. [Kaggle](https://www.kaggle.com/) の公式データを `data/` フォルダに配置する

3. ライブラリをインストールする：

   ```bash
   pip install -r requirements.txt
   ```

4. `notebooks/` 以下のファイルを上から順に実行する

---

## ⚠️ ライセンスについて

現在、このプロジェクトは他者コードの参照を含む可能性があるため、**明示的なライセンスは設定していません**。再利用の際は該当コードの出典とライセンスをご確認ください。

---
