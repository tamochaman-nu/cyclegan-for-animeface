# CycleGAN (for AnimeFace)

## 概要

このプロジェクトは、ペアとなる画像データが存在しない場合でも画像から画像への変換 (Image-to-Image Translation) を学習できる、Cycle-Consistent Adversarial Networks (CycleGAN) のTorch実装ベースのプロジェクトです。
本プロジェクトではDockerおよびDocker Composeを利用しており、ホスト環境の依存関係を気にすることなく、GPUを活用した学習や推論（テスト）の環境を簡単に構築・実行できるように整備されています。

## 前提条件

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2推奨: `docker compose` コマンド)
- NVIDIA GPU および [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPUでの学習・推論を行う場合)

## データセットの準備

独自のデータセット（例：アニメ顔データなど）でCycleGANを学習させるには、プロジェクト内の `datasets` フォルダに以下のようなディレクトリ構成で画像を配置する必要があります。

```text
datasets/
└── <データセット名>
    ├── trainA   # ドメインAの学習用画像
    ├── trainB   # ドメインBの学習用画像
    ├── testA    # ドメインAのテスト（推論）用画像
    └── testB    # ドメインBのテスト（推論）用画像
```

### 既存データセットのフォーマット（シンボリックリンク作成）

NASや別ドライブに既に大量の画像データがドメインごとに存在する場合、データを直接コピーせずにシンボリックリンクを用いて上記のディレクトリ構造を生成するスクリプト `prepare_dataset.py` が利用できます。

```bash
python3 prepare_dataset.py --dir_A /path/to/domain_A_images --dir_B /path/to/domain_B_images --dataset_name <作成するデータセット名>
```

**主な引数**:
- `--dir_A`: ドメインAの元画像が含まれるディレクトリです。サブフォルダ内の画像も再帰的に検索されます。
- `--dir_B`: ドメインBの元画像が含まれるディレクトリです。
- `--dataset_name`: 新しく作成されるデータセットの名前です。
- `--output_root`: データセットの出力先ルートディレクトリ（デフォルト: `./datasets`）
- `--test_ratio`: テスト用に分割する画像の割合（デフォルト: `0.1`＝10%）

*注意：ドメインAとドメインBでファイルの収集数が異なる場合、スクリプトは自動的に**少ない方の枚数に合わせて**両方のドメインのファイルリストをランダムに均等化（切り詰め）した上で学習/テスト用の分割を行います。*

このスクリプトを実行すると、指定した `--output_root` 配下にシンボリックリンクが張られた `trainA`, `trainB`, `testA`, `testB` ディレクトリが生成され、すぐにCycleGANの学習データとして使用できる状態になります。
*※Windows環境などで権限が不足しておりシンボリックリンクの作成に失敗した場合は、自動的に実ファイルのコピーが行われます。*

> [!WARNING]
> WindowsホストからDockerを利用して学習を行う場合、`prepare_dataset.py` で作成されたWindows絶対パス表記のシンボリックリンクがLinuxコンテナ内で解決できず `FileNotFoundError` になる問題があります。  
> WindowsでDockerを利用される方は、シンボリックリンク機能を使用せず実ファイルのコピーを行う **`prepare_dataset_windows.py`** を代わりに実行してデータセットを作成してください（引数等の使い方は同じです）。

## 実行方法

本プロジェクトでは `docker-compose.yml` に学習用の `train` サービスと、推論用の `inferance` サービスが定義されています。引数はコマンドライン引数としてコンテナ内のPythonスクリプト( `train.py` / `test.py` ) に渡されます。

### 学習 (Training)

モデルの学習を開始するには、以下のDocker Composeコマンドを実行します。

```bash
docker compose run --rm train --dataroot ./datasets/<データセット名> --name <モデル名>
```

- `--dataroot`: データセットが格納されているディレクトリのパスを指定します。
- `--name`: 学習するモデルの名前（プロジェクトにおける実験名）を指定します。学習経過やチェックポイントはこの名前で `./checkpoints/<モデル名>` セクション等に保存されます。

**学習状況のモニタリング (TensorBoard)**:
学習中の損失(Loss)などの推移はTensorBoardで確認できます。学習コマンドとは**別のターミナル**を開き、以下のコマンドでTensorBoardコンテナをバックグラウンド起動してください。
```bash
docker compose up -d tensorboard
```
起動後、ブラウザから `http://localhost:16006` にアクセスして学習の経過を確認できます。確認が終わったら `docker compose stop tensorboard` で停止できます。

**CPUを利用して学習する場合（GPU非搭載環境）**:
コマンドの引数に `--gpu_ids -1` を追加して実行します。
```bash
docker compose run --rm train --dataroot ./datasets/<データセット名> --name <モデル名> --gpu_ids -1
```

*※ 学習・推論を開始する際、GPU・CPUのどちらで実行されているか（およびGPUのデバイス名）が自動的にコンソールへログ出力されるため、正しく設定できているか確認できます。*

### 推論 / テスト (Inference)

学習済みモデルを使用して画像変換（推論）を行うには、以下のコマンドを実行します。

```bash
docker compose run --rm inferance --dataroot ./datasets/<データセット名> --name <モデル名>
```

- 推論結果の画像は `./results/<モデル名>/test_latest/index.html` にWebページ形式で出力されるため、ブラウザから簡単に結果を確認することができます。
- **片方向のみの推論**を行う場合は、引数で方向を `--direction AtoB` または `--direction BtoA` として指定します。

```bash
docker compose run --rm inferance --dataroot ./datasets/<データセット名> --name <モデル名> --direction AtoB
```

## カスタマイズ・オプション項目

画像サイズ、バッチサイズ、学習率といったより詳細な学習および推論時の設定については、プロジェクト内の `options.py` ファイルをご確認ください。
同ファイル内に `argparse` で設定可能な各パラメーターとそのデフォルト値が記載されています（例: リサイズ設定 `--preprocess` など）。

### ハイパーパラメータチューニング (Optuna)

Optunaを利用して学習率や損失の重み（Cycle loss, Identity loss, Perceptual loss）、残差ブロック数などのハイパーパラメータを自動探索することができます。

1. **最適化の実行**:
   以下のコマンドでOptunaの探索を実行します。
   ```bash
   docker compose run --rm optuna_train --dataroot ./datasets/<データセット名> --name <モデル名> --optuna_trials 50 --n_epochs_optuna 10
   ```
   - `--optuna_trials`: 実行するトライアルの回数（デフォルト: 50）
   - `--n_epochs_optuna`: 1つのトライアルあたりのエポック数（デフォルト: 10）

2. **探索結果の確認 (Optuna Dashboard)**:
   最適化の進捗や各ハイパーパラメータの重要度などは、ダッシュボードから確認できます。
   ```bash
   docker compose up -d optuna_dashboard
   ```
   起動後、ブラウザから `http://localhost:8080` にアクセスしてください。

3. **各トライアルの損失推移の確認 (TensorBoard)**:
   各トライアルごとの損失推移はTensorBoardで確認できます。各トライアルは `<モデル名>_trial_0` のような名前で記録されます。
   ```bash
   docker compose up -d tensorboard
   ```
   ブラウザから `http://localhost:16006` にアクセスしてください。
