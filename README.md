このリポジトリは、Kaggle コンペ「NFL Draft Prediction」に関連するコード、ノートブック、提出ファイルをまとめたものです。

## /


## 📁 notebook 目次

<details>
<summary><strong>00_baseline (0.80792).ipynb</strong></summary>

ベースラインモデル。  
・欠損値補完はすべて平均値  
・特徴量に BMI を追加  
・モデルはランダムフォレスト

</details>

<details>
<summary><strong>01_0620_preprocessing(未提出).ipynb</strong></summary>

主に EDA を行ったファイル  
・欠損値を補完せずに削除した  
・スコアは大幅に下がった

</details>

<details>
<summary><strong>02_0621_notebook (0.79804).ipynb</strong></summary>

ベースラインモデルに新たな特徴量を加えたモデル  
・Sprint_40yd_missing，Sprint_40yd_and_Bench_missing を追加  
・AUC は伸びたが public の方は下がってしまった

</details>

<details>
<summary><strong>03_0624_notebook(未提出)</strong></summary>

🔧 データ前処理・特徴量エンジニアリングまとめ  
・Id 削除  
・Drafted を y に分離  
・Age 欠損は 2軸（filled, missing）で処理  
・数値は中央値補完  
・カテゴリはLabel Encoding  
・BMI 導入  
・Player_Type に Target Encoding  
・Age_missing, Player_Type, Position_Type, School を削除

⚙️ モデル構築  
・LightGBM（過学習対策多数）  
・5-fold CV + AUC + EarlyStopping(50)

📈 評価結果（例）  
・Train AUC：0.9166  
・Valid AUC：0.8254

</details>
<details> <summary><strong>04_0626_notebook (未提出)</strong></summary>
📊 特徴量の精査とLightGBMの最適化

・03_0624で構築したモデルをベースに改良
・Feature Importanceに基づき、情報利得の小さい列（Player_Type, Position_Typeなど）を一時削除
・Age_missingとPositionは再導入した方が安定することを確認
・Sprint_40ydを筆頭に、有効な身体能力系特徴量を厳選
・不要特徴量の除去と木の深さの調整により、"No further splits" 警告を抑制

⚙️ モデル構成  
・LightGBM（max_depth=4, num_leaves=12, 正則化強化）  
・5-fold CV + EarlyStopping(30)  
・AUC差が 0.05 以下になるよう精密に調整  

📈 評価結果（最終）  
・Average Train AUC：0.8693  
・Average Validation AUC：0.8216  
・差分：0.0477（過学習抑制に成功）  

✅ 最終モデル構成（提出候補）：
```python
model = LGBMClassifier(
    max_depth=4,
    num_leaves=12,
    min_child_samples=30,
    min_split_gain=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=3.0,
    reg_lambda=2.0,
    learning_rate=0.05,
    n_estimators=500,
    random_state=42
)
```

</details>

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
