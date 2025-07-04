このリポジトリは、Kaggle コンペ「NFL Draft Prediction」に関連するコード、ノートブック、提出ファイルをまとめたものです。
0707まで

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
<details> <summary><strong>09_0627_notebook (0.84205)</strong></summary>
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
<details> <summary><strong>10_0627_notebook (0.83668)</strong></summary>
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

<details> <summary><strong>11_0628_notebook (0.8208)</strong></summary>
📊 不要特徴量削除 + Optuna 最適化による最高スコア更新モデル

・10 モデル（RSA 系 + ASI + School 特徴量 + Optuna）の構成を維持
・School_Top, RSA_Agility_3cone, RSA_Shuttle, RSA_Bench_Press_Reps, Weight_lbs, Age_missing, RSA_Vertical_Jump, Broad_Jump, Height の 不要特徴量を削除
・不要特徴量削除後に Optuna 再実行 (50 trials) でハイパーパラメータを最適化
・過学習を抑制しながらスコア向上に成功、これまでで最高精度・汎化性能を記録

⚙️ モデル構成

・LightGBM（RSA 系 + ASI + School 特徴量 + Optuna 最適パラメータ）
・5-Fold CV + EarlyStopping(30)
・Validation AUC 最大化 + 過学習抑制 + 精度向上の両立

📈 評価結果（最終）

・Average Train AUC：0.9434
・Average Validation AUC：0.8524
・差分：0.0910（適度な差で汎化性能も担保、過去最高スコア）

✅ 最終モデル構成（提出モデル）

```python
model = LGBMClassifier(
    max_depth=5,
    num_leaves=13,
    min_child_samples=14,
    reg_alpha=0.070,
    reg_lambda=0.034,
    learning_rate=0.100,
    n_estimators=1000,
    subsample=0.820,
    colsample_bytree=0.665,
    random_state=42
)
```

✅ 不要特徴量削除 + Optuna により Validation AUC を 0.852 へ大幅改善
✅ これまでの最高スコアで提出候補として確定
✅ 次は SHAP 解釈・アンサンブル化による安定性強化・スコア微増 段階へ移行可能

</details>

<details> <summary><strong>12_0629_notebook (0.82986)</strong></summary>

</details>

<details> <summary><strong>13_0630_notebook (0.8366)</strong></summary>
📊 最適特徴量削除 + SHAP 解析による重要特徴量確定 + 安定化モデル（提出候補）

✅ 概要
11 モデル（RSA 系 + ASI + School 特徴量 + Optuna）構成を踏襲

SHAP 解析により有効特徴量・不要特徴量を再整理

不要特徴量を削除し、モデルのシンプル化・安定化を実現

School_Drafted_Rate_TE にスムージング Target Encoding を適用し情報量確保＆リーク防止

過去最高水準の Validation AUC を達成しつつ AUC 差も適度に抑制

Public AUC 0.8495 を達成（0.85 目前）

⚙️ モデル構成
LightGBM（SHAP 解析で確定した有効特徴量 + 最適パラメータ）

5-Fold Stratified CV + EarlyStopping(30)

Validation AUC 最大化 + 安定性確保

📈 評価結果（最終）
指標 スコア
Average Train AUC 0.9356
Average Validation AUC 0.8495
差分 0.0861

✅ 過学習を抑えつつ高精度・安定性を維持し提出候補水準に到達

✅ 特徴量
使用特徴量：

Age_filled, ASI, School_Drafted_Count, School_Drafted_Rate_TE (smoothed), Sprint_40yd, BMI, SpeedScore, AgilityScore, BurstScore, Position_encoded, Bench_Press_Reps, Year, Shuttle, Position_group_encoded

削除した特徴量（SHAP/Feature Importance 解析に基づき無効・ノイズと判断）：

Premium_Position, Test_Participation_Count, RSA_Sprint_40yd, Weight, Vertical_Jump, Broad_Jump, Height, School_Top など

✅ モデルパラメータ（提出モデル）

```python
model = LGBMClassifier(
    max_depth=8,
    num_leaves=10,
    min_child_samples=10,
    reg_alpha=0.0415,
    reg_lambda=0.2428,
    learning_rate=0.0726,
    n_estimators=1000,
    subsample=0.6898,
    colsample_bytree=0.7463,
    random_state=42
)
```

</details>

<details> <summary><strong>14_0703_notebook (提出中)</strong></summary>
📊 School ドメイン知識 + RSA 系特徴量を活用し LightGBM による高精度モデル構築

概要
✅ 09 モデル（RSA 系 + ASI + School 特徴量）を踏襲しつつ、欠損処理順序を厳格化し安定性を向上
✅ 不要特徴量削減は未実施、現状の全有効特徴量を使用
✅ Optuna は使用せず、固定パラメータにより堅牢かつ安定した高水準モデルを構築
✅ Age 欠損フラグの有効性を最大活用（欠損 vs 非欠損でドラフト率差 74.7%）

⚙️ モデル構成
LightGBM（RSA 系 + ASI + School 特徴量）

パラメータは 09 構成を踏襲（Optuna 無し、固定）

5-Fold CV + EarlyStopping(30)

Validation AUC 最大化 + 過学習抑制

```python
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

📈 評価結果（最終）
Average Train AUC：0.8980

Average Validation AUC：0.8538

差分：0.0442（安定した汎化性能と高スコアを両立）

✅ 特筆事項
Age 欠損処理の適正化で精度向上

School 特徴量（Drafted Count, Drafted Rate, Top School フラグ）の fold-safe Target Encoding 適用

RSA 特徴量・ASI 特徴量を維持活用

Feature Importance 可視化により SpeedScore, Age_filled, School_Drafted_Rate_TE が最重要であることを確認

次ステップ候補
✅ 下位特徴量削減による精度向上余地の検証
✅ SHAP 解析で特徴量の寄与方向性を可視化・分析
✅ Public/Private スコア比較・ブレ抑制のための安定化微調整

0.85 台を安定して記録可能な提出候補モデル完成。
必要に応じて次回は特徴量削減・SHAP 解析・最終提出調整へ進行可能。

</details>

<details> <summary><strong>15_0703_notebook (提出準備中)</strong></summary>
📊 Optuna 最終調整・特徴量洗練による安定高精度モデル

09/10 モデルの RSA 系・ASI・School 特徴量構成を維持しつつ、不要特徴量を段階的に削除

SHAP 解析を用いリーク疑い・過剰寄与特徴量を削減

Optuna (50 trials) により max_depth, num_leaves, min_child_samples, reg_alpha, reg_lambda, learning_rate を微調整

最終的に分布を絞り精度向上を目指した結果、上振れの可能性が高いモデル構成を完成

⚙️ モデル構成

LightGBM（Optuna 微調整済）

5-Fold Stratified CV + EarlyStopping(30)

Validation AUC 最大化を重視

Drafted 平均：0.637 （高スコアモデルに完全一致）

Drafted 標準偏差：0.202（低め、過信モデル寄り）

📈 評価結果（最終）

Average Train AUC：0.89〜0.90

Average Validation AUC：0.84 前後（Public 0.84 超えを狙う）

✅ 最適化結果（Best Params）

```python
model = LGBMClassifier(
    max_depth=4,
    num_leaves=10,
    min_child_samples=20,
    reg_alpha=5.85,
    reg_lambda=6.01,
    learning_rate=0.027,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

🛠️ ポイント
Drafted 分布の平均を過去高スコアモデルと揃えつつ、より確信度の高い予測を実現

Optuna によるパラメータ微調整で汎化性能を最大化

さらなるスコア向上が見込める場合は：

Std を適度に上げる（分布を散らす）

SHAP 解析で寄与度低い特徴量の再整理

</details>
<details> <summary><strong>16_0703_notebook (提出中)</strong></summary>
📊 RSA / ASI / School特徴量構成維持 × Optuna重み最適化Votingモデル

・09, 10 モデル（RSA 系 + ASI + School 特徴量 + LightGBM 最適化）を踏襲
・Optuna により VotingClassifier (LGBM, CatBoost, XGBoost) の重み最適化 を実施
・CatBoost / XGBoost 重視の重み配分で Public スコア最大化を狙う
・Public 提出前の予測分布類似度分析で 09 モデルに極めて近い分布（L1 距離: 4.29）を確認済み

⚙️ モデル構成

・LightGBM / CatBoost / XGBoost による soft voting
・Optuna 最適化後の重み設定：

python
コードをコピーする
weights = [1.479, 4.943, 4.988] # [LGBM, CatBoost, XGBoost]
・5-Fold Stratified CV + EarlyStopping(30)
・Validation AUC に基づく最適化

⚙️ 特徴量構成

RSA 系特徴量 (Sprint_40yd, Vertical_Jump, Bench_Press_Reps, Shuttle, Agility_3cone)

ASI (Athletic Score Index)

School 特徴量 (Drafted Count, Drafted Rate, Top School Flag)

Position_encoded, Position_group_encoded

Age_filled, Age_missing

BMI, SpeedScore, BurstScore, AgilityScore

📈 評価結果（最終）

・Average Train AUC：0.9851
・Average Validation AUC：0.8446

Fold 別：

Fold 1: 0.8221

Fold 2: 0.8716

Fold 3: 0.8576

Fold 4: 0.8225

Fold 5: 0.8492

✅ Public 提出時に 0.847〜0.849 程度を期待できる水準

✅ 特徴・位置付け

Optuna による Voting 重み最適化完了モデル

RSA/ASI/School 特徴量の最適化構成維持

Public 上位狙い提出用として完成形

</details>

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
