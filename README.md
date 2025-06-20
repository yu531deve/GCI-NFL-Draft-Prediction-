
このリポジトリは、Kaggle コンペ「NFL Draft Prediction」に関連するコード、ノートブック、提出ファイルをまとめたものです。

---

## 📁 プロジェクト構成

<details>
<summary>▼ クリックして展開</summary>

```
nfl-draft-prediction/
├── data/                  # Kaggle 公式データ（.gitignoreで除外）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/             # 分析・実験用のノートブック
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_lgb_optuna.ipynb
│   └── 05_stack_ensemble.ipynb
├── src/                   # 再利用可能なスクリプト・関数類
│   ├── features.py
│   ├── model.py
│   ├── utils.py
│   └── config.py
├── submissions/           # 提出ファイルの保存場所
│   ├── baseline.csv
│   ├── lgb_optuna.csv
│   └── ensemble.csv
├── models/                # 学習済みモデル（.gitignoreで除外）
│   ├── model_lgb.pkl
│   └── encoder.pkl
├── output/                # グラフや分析結果の出力（任意）
│   ├── feature_importance.png
│   └── correlation_matrix.png
├── README.md              # 本ファイル
├── requirements.txt       # 使用ライブラリ一覧（pip freeze 出力）
├── .gitignore             # 除外対象の定義（data/, models/など）
└── LICENSE                # ライセンス（MITなど、※現在は未定）
```

</details>


---

## 📓 ノートブック命名ルール

| ファイル名                  | 内容                                      |
|---------------------------|-------------------------------------------|
| `01_eda.ipynb`            | 初期の可視化と仮説立案                     |
| `02_preprocessing.ipynb`  | 欠損値補完、カテゴリ変数の処理など         |
| `03_model_baseline.ipynb` | ランダムフォレストなどによる初期モデル     |
| `04_lgb_optuna.ipynb`     | LightGBM + Optuna によるチューニング       |
| `05_stack_ensemble.ipynb` | スタッキングやアンサンブルの実験           |

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
