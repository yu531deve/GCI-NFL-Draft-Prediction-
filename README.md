NFL Draft Prediction - Kaggle Competition
このリポジトリは、Kaggle コンペ「NFL Draft Prediction」に関連するコード、ノートブック、提出ファイルをまとめたものです。

プロジェクト構成
nfl-draft-prediction/
├── data/ Kaggle 公式データ（.gitignore で除外）
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
├── notebooks/ 分析・実験用のノートブック
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_baseline.ipynb
│ ├── 04_lgb_optuna.ipynb
│ └── 05_stack_ensemble.ipynb
├── src/ 再利用可能なスクリプト・関数類
│ ├── features.py
│ ├── model.py
│ ├── utils.py
│ └── config.py
├── submissions/ 提出ファイルの保存場所
│ ├── baseline.csv
│ ├── lgb_optuna.csv
│ └── ensemble.csv
├── models/ 学習済みモデル（.gitignore で除外）
│ ├── model_lgb.pkl
│ └── encoder.pkl
├── output/ グラフや分析結果の出力（任意）
│ ├── feature_importance.png
│ └── correlation_matrix.png
├── README.md 本ファイル
├── requirements.txt 使用ライブラリ一覧（pip freeze 出力）
├── .gitignore 除外対象の定義（data/, models/など）
└── LICENSE ライセンス（MIT など）

ノートブック命名ルール
01_eda.ipynb
初期の可視化と仮説立案

02_preprocessing.ipynb
欠損値補完、カテゴリ変数の処理など

03_model_baseline.ipynb
ランダムフォレストなどによる初期モデル

04_lgb_optuna.ipynb
LightGBM + Optuna によるチューニング

05_stack_ensemble.ipynb
スタッキングやアンサンブルの実験

ファイルの分類方針
data/, models/: 頻繁に変化するローカルデータ（.gitignore で除外）

notebooks/: ノートブックによる実験記録（ステップ順に命名）

src/: 再利用コード（.py に整理して import）

submissions/, output/: 提出物や図などのアウトプット

README.md, requirements.txt: プロジェクト構成情報と環境再現用

使用方法
リポジトリをクローンする

Kaggle の公式データを data/フォルダに配置する

以下を実行して必要なライブラリをインストールする

pip install -r requirements.txt

notebooks 以下のファイルを上から順に実行する

ライセンス
このプロジェクトは MIT ライセンスの下で公開されています。詳細は LICENSE ファイルを参照してください。
