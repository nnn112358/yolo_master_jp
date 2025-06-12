# 2-Beverages-Labeling

本章チュートリアルでは、選択した[魚眼レンズ_スマート販売データセット](https://aistudio.baidu.com/datasetdetail/91732/0)と[飲物データセット](https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Drink_284_Detection_Dataset/Drink_284_Detection_YOLO.zip)の2つのデータセットをサンプルデータセットとして、プライベートデータセットのさらなる処理と自動ラベリングの方法を手取り足取り教えます。以下が主要なステップと効果のスクリーンショットです
- フォーマット変換処理（VOC2YOLO）
- データセットの結合とサンプリング
- カスタムモデルの訓練（YOLOv8m）
- ラベリングツール X-AnyLabeling と組み合わせた自動ラベリング

![image](https://github.com/user-attachments/assets/3f46b897-1fee-4ac3-8a2f-92538ba60e32)

![image](https://github.com/user-attachments/assets/6ec7848b-fb39-4087-9c86-c5b577fd2a25)


## プロジェクト構造
```python
2-Beverages-Labeling/

├── 00-dataset                          # タスクデータセットの紹介と準備
│   └── VOC_to_YOLO.py                    # # VOC フォーマットから YOLO フォーマットへの変換スクリプト
├── 01-data_merging                     # データセットの結合
│   ├── dataset_merge.py                  # # 複数データセット結合スクリプト
│   └── label_id_update.py                # # ラベル ID 更新マッピングスクリプト 
├── 02-ulralytics_based_model_training  # ultralytics ベースのカスタムモデル訓練、X-AnyLabeling とファインチューンモデルを使用してプライベートデータセットにラベル付け
│   ├── data.yaml                         # # データセット配置ファイル
│   ├── train.py                          # # モデル訓練スクリプト
│   └── export.py                         # # モデルエクスポートスクリプト
└── README.md
```


## プログラムファイル詳細説明

### タスクデータセットの紹介と準備

- **[VOC_to_YOLO.py](./00-dataset/VOC_to_YOLO.py)**：**このスクリプトはVOCフォーマットのデータセットをYOLOフォーマットに変換するためのものです。** 実際のアプリケーションでは、異なるソースのデータセットは異なるアノテーションフォーマットを使用している場合があり、VOCは一般的なフォーマットの1つであり、YOLO訓練には特定のフォーマットが必要であるため、このスクリプトがフォーマット変換の問題を解決します。

### データセットの結合

![image](https://github.com/user-attachments/assets/06e117c5-c651-43fa-bdad-60c97e3bb147)

![image](https://github.com/user-attachments/assets/7e5801a9-fe8f-4ce2-b890-76882276e750)


- **[dataset_merge.py](./01-data_merging/dataset_merge.py): 2つの異なる飲物データセットの結合**（*Drink_284_Detection_Labelme*と*魚眼レンズ_スマート販売データセット*）、カテゴリ重複問題の処理、データセットのバランスの保証。スクリプトは重複カテゴリを含むサンプルを優先的に選択し、モデルがこれらの共有特徴を学習できることを保証します。

- **[label_id_update.py](./01-data_merging/label_id_update.py): ラベル ID の更新**、異なるデータセットのラベル ID を統一。スクリプトはインデックスマッピング辞書を作成し、これに基づいてすべてのラベルファイル内のカテゴリインデックスを更新します。

### ultralyticsベースのカスタムモデル訓練とONNXエクスポート。

- **[data.yaml](./02-ulralytics_based_model_training/data.yaml)**: データセット配置ファイル。訓練と検証データのパス、およびすべてのカテゴリ名（合計113カテゴリ）を定義し、YOLOv8m訓練時にこのファイルを通じてデータセットの関連情報を取得します。

- **[train.py](./02-ulralytics_based_model_training/train.py)**: モデル訓練スクリプト。事前訓練済みYOLOv8m重みをファインチューンして転移学習を行い、エポック数、画像サイズ、バッチサイズなどのさまざまな訓練パラメータを設定できます。

- **[export.py](./02-ulralytics_based_model_training/export.py)**: ファインチューン後のモデルをONNXフォーマットでエクスポートし、プライベートデータセットのラベリングに使用します。

## 飛書チュートリアル

本章チュートリアルで選択したサンプルデータセットは*魚眼レンズ_スマート販売データセット*と*飲物データセット*の2つのデータセットから来ており、2つのデータセットに対してフォーマット変換処理、結合、サンプリングを行い、プライベートデータセットのさらなる処理方法を手取り足取り教え、カスタムモデル（YOLOv8m）を訓練し、ラベリングツールX-AnyLabelingと組み合わせて自動ラベリングを行います。


**チュートリアルURL**：https://wvet00aj34c.feishu.cn/docx/R04QdmQMMoaA44xyDYkcA0AfnOd

以下は簡単な使用方法です。詳細は飛書チュートリアルをご確認ください。

### 00. データセットフォーマット変換

`./00-dataset/VOC_to_YOLO.py`を実行し、VOCフォーマットのデータセットをYOLOフォーマットに変換します。

### 01. データセット結合とラベル更新

1. `./01-data_merging/dataset_merge.py`を実行し、2つの異なる飲物データセットを結合し、カテゴリ重複問題を処理し、データセットのバランスを保証します。

2. `./01-data_merging/label_id_update.py`を実行し、ラベルIDを更新し、異なるデータセットのラベルIDを統一します。

### 02. モデル訓練とエクスポート

1. `./02-ulralytics_based_model_training/train.py`を実行し、事前訓練済みYOLOv8m重みを使用して転移学習を行い、必要に応じて`./02-ulralytics_based_model_training/data.yaml`の訓練パラメータを調整します。

2. `./02-ulralytics_based_model_training/export.py`を実行し、訓練済みモデルをONNXフォーマットでエクスポートします。

### 03. プライベートデータセットラベリング

- **X-AnyLabelingツールとエクスポートしたONNXモデルを使用してプライベートデータセットにラベル付け。** 具体的な操作は[**飛書チュートリアル**](https://wvet00aj34c.feishu.cn/docx/R04QdmQMMoaA44xyDYkcA0AfnOd)をご参照ください。

## データセットと訓練済みモデルのダウンロード

> YOLO Masterは魔塔コミュニティにYOLO Masterプロジェクト専用のModelScope組織を設立し、YOLO各バージョンの公式モデル重みをここにバックアップし、国内の学習者が使いやすくし、コントリビューターが容量の大きい実験結果や重みを保存できるようサポートします。

> YOLO Master ModelScope 組織アドレス：https://modelscope.cn/organization/yolo_master

本チュートリアルのデータセット、モデルは **[YOLO実践のデータセット結合と自動ラベリングチュートリアルファイルコレクション](https://www.modelscope.cn/collections/YOLO-shijianzhishujujihebingyuzi-54143625553f44)** に保存されています。学習者は上記リンクを訪問して必要なリソースを取得できます。

![image](https://github.com/user-attachments/assets/f039c2c6-c642-4ce2-a13e-0f57ad7c70d2)

![image](https://github.com/user-attachments/assets/847bec8e-c842-4a33-83a4-f5e52c03f8be)


