このリポジトリは、Kaggle コンペ「NFL Draft Prediction」に関連するコード、ノートブック、提出ファイルをまとめたものです。

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
・Age 欠損は 2 軸（filled, missing）で処理  
・数値は中央値補完  
・カテゴリは Label Encoding  
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
<details> <summary><strong>04_0626_notebook (0.82782)</strong></summary>
📊 特徴量の精査とLightGBMの最適化

・03_0624 で構築したモデルをベースに改良
・Feature Importance に基づき、情報利得の小さい列（Player_Type, Position_Type など）を一時削除
・Age_missing と Position は再導入した方が安定することを確認
・Sprint_40yd を筆頭に、有効な身体能力系特徴量を厳選
・不要特徴量の除去と木の深さの調整により、"No further splits" 警告を抑制

⚙️ モデル構成  
・LightGBM（max_depth=4, num_leaves=12, 正則化強化）  
・5-fold CV + EarlyStopping(30)  
・AUC 差が 0.05 以下になるよう精密に調整

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
<details> <summary><strong>05_0626_notebook (0.82769)</strong></summary>
📊 ポジション情報の強化とOptunaによる自動チューニング

・Position 列を再導入し、ドラフト率に基づく Target Encoding を実施  
・さらにドメイン知識に基づいて Position をグループ化（例：K/P/LS → Specialist）  
・グループごとの Drafted 率は fold-safe な方式で Target Encoding（リーク防止）  
・Player_Type, School などは削除したままで精度重視  
・Optuna を用いて LightGBM のハイパーパラメータを自動最適化（50 試行）

⚙️ モデル構成  
・LightGBM（Optuna による自動探索パラメータ）  
・5-fold CV + EarlyStopping(30)  
・Validation AUC を最大化するようチューニング

📈 評価結果（最終）  
・Average Train AUC：0.8972  
・Average Validation AUC：0.8303  
・差分：0.0669（やや過学習傾向だが許容範囲）

✅ 最終モデル構成（提出候補）：

```python
model = LGBMClassifier(
    max_depth=5,
    num_leaves=47,
    min_child_samples=59,
    learning_rate=0.06596,
    subsample=0.6411,
    colsample_bytree=0.7170,
    reg_alpha=0.4877,
    reg_lambda=7.7297,
    n_estimators=700,
    random_state=42
)
```

</details>

<details> <summary><strong>06_0627_notebook (0.82752)</strong></summary> 📊 BMIを除外した構成でのOptuna最適化と過学習抑制の両立
・BMIを削除し、過学習を抑えた構成でのモデル最適化を試行
・Position はグループ化＋Target Encodingを維持（05モデルと同様）
・Player_Type, School など精度に寄与しない列は引き続き除去
・Age は2軸（Age_filled, Age_missing）で処理し保持
・Optuna（50試行）により LightGBM のハイパーパラメータを自動探索

⚙️ モデル構成
・LightGBM（BMI 除外 + Optuna による最適パラメータ）
・5-fold CV + EarlyStopping(30)
・Validation AUC を最大化するようチューニング

📈 評価結果（最終）
・Average Train AUC：0.8803
・Average Validation AUC：0.8327
・差分：0.0476（05 モデルより過学習が抑制され、精度も向上）

✅ 最終モデル構成（提出候補）：

```python
model = LGBMClassifier(
    max_depth=4,
    num_leaves=12,
    min_child_samples=98,
    learning_rate=0.07784724324991651,
    n_estimators=700,
    subsample=0.5050379002287039,
    colsample_bytree=0.50027338347916,
    reg_alpha=3.037811473368862,
    reg_lambda=3.294160938150066,
    random_state=42
)
```

</details>

<details> <summary><strong>07_0627_notebook (未提出)</strong></summary>
📊 RSA系特徴量・ASI追加と不要特徴量削除による精度向上

・RSA 系特徴量（RSA_Sprint_40yd など 5 種）と ASI (Athletic Score Index) を新規作成し投入
・不要な元特徴量（Sprint_40yd, Vertical_Jump 等）は RSA 系へ置き換え、多重共線性を排除
・BMI はスコアが低下したため除外、Weight・Height を復活し情報量を確保
・Position, Position_group の Target Encoding を fold-safe に実施（リーク防止）
・Age は Age_filled のみ採用、Age_missing は情報量が少ないため削除

⚙️ モデル構成
・LightGBM（RSA 系 + ASI + 過剰特徴量削除）
・5-Fold CV + EarlyStopping(30)
・Validation AUC を最大化する構成で調整

📈 評価結果（最終）
・Average Train AUC：0.8755
・Average Validation AUC：0.8349
・差分：0.0406（安定した汎化性能で提出候補レベル）

✅ 最終モデル構成（提出候補）

```
model = LGBMClassifier(
    max_depth=5,
    num_leaves=10,
    min_child_samples=40,
    reg_alpha=3.0,
    reg_lambda=3.0,
    learning_rate=0.02,
    n_estimators=900,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

```

</details>

<details> <summary><strong>08_0627_notebook (0.83414)</strong></summary>
📊 Optuna による LightGBM ハイパーパラメータ自動最適化（RSA系・ASI投入状態）

・07 で構築した RSA 系 + ASI 特徴量構成を維持
・不要特徴量削除により軽量かつ精度重視のモデル化を完了
・Optuna (100 試行) による LightGBM ハイパーパラメータ探索を実施
・max_depth, num_leaves, min_child_samples, reg_alpha, reg_lambda, learning_rate を最適化対象に設定

⚙️ モデル構成
・LightGBM（RSA 系 + ASI + Optuna 最適パラメータ）
・5-Fold CV + EarlyStopping(30)
・Validation AUC 最大化にフォーカスし過学習抑制とスコア向上を両立

📈 評価結果（最終）
・Average Train AUC：0.8823
・Average Validation AUC：0.8377
・差分：0.0446（過去最高水準のスコア、提出準備完了）

✅ 最適化結果（Best Params）

```
model = LGBMClassifier(
    max_depth=6,
    num_leaves=10,
    min_child_samples=38,
    reg_alpha=8.18,
    reg_lambda=8.07,
    learning_rate=0.0442,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

```

</details>
<details> <summary><strong>09_0627_notebook (提出中)</strong></summary>
📊 School ドメイン知識活用によるスコア向上

・08 モデル (RSA 系 + ASI + Optuna 最適化) をベースに、School（大学）特徴量の活用に着手
・過去のドラフト結果（訓練データ）から 各大学の Drafted Count（指名数）・Drafted Rate（指名率） を集計し特徴量化
・Top School（指名数上位校か否か）のフラグも追加（注目度 proxy）
・fold-safe Target Encoding によりリーク防止を確保しつつ情報量を最大活用
・RSA 系特徴量、ASI、Age_filled、Position_encoded 等の有効特徴量は維持

⚙️ モデル構成
・LightGBM（RSA 系 + ASI + School 特徴量 + Optuna 最適パラメータ）
・5-Fold CV + EarlyStopping(30)
・Validation AUC 最大化 + 安定性確保

📈 評価結果（最終）
・Average Train AUC：0.8937
・Average Validation AUC：0.8384
・差分：0.0553（安定した汎化性能で過去最高水準、提出候補レベル）

✅ 最終モデル構成（提出候補）

```
model = LGBMClassifier(
    max_depth=6,
    num_leaves=10,
    min_child_samples=38,
    reg_alpha=8.18,
    reg_lambda=8.07,
    learning_rate=0.0442,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

```

✅ School 特徴量導入で Validation AUC を 0.838 台に向上
✅ さらなる微調整・Feature Selection・Optuna 再実行で 0.840 超えを狙う準備段階

</details>
<details> <summary><strong>10_0627_notebook (提出予定)</strong></summary>
📊 Optuna による最終 LightGBM 最適化・スコア最大化モデル

・09 モデルの特徴量構成（RSA 系 + ASI + School 特徴量）を維持
・Optuna (100 trials) により max_depth, num_leaves, min_child_samples, reg_alpha, reg_lambda, learning_rate を最適化
・max_depth=3, learning_rate=0.087 と浅め・速めの収束で高精度化＆汎化性能向上
・スコアはこれまでの最高値を記録

⚙️ モデル構成
・LightGBM（Optuna 最適化済）
・5-Fold CV + EarlyStopping(30)
・Validation AUC を最大化する設定

📈 評価結果（最終）
・Average Train AUC：0.89〜0.90（予定）
・Average Validation AUC：0.85 前後（予定）

✅ 最適化結果（Best Params）

```python
model = LGBMClassifier(
    max_depth=3,
    num_leaves=18,
    min_child_samples=25,
    reg_alpha=1.17,
    reg_lambda=4.84,
    learning_rate=0.087,
    n_estimators=1200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

## </details>

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
│ └── \*.csv
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

````

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
````
