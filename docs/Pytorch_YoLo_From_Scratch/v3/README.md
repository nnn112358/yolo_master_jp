# YoLo Master

## YOLO Models from Scratch

本プロジェクトは**手動でYOLOモデルを実装する**方式で、**小規模データセット上でYOLOアルゴリズムを再現**することを目的としています。現在の計画ではYOLO [**V1**](../v1/YOLOv1.ipynb)、[**V3**](../v3/YOLOv3.ipynb)、[**V5**](../v5/YOLOv5.ipynb)、V8などを手動で実装する予定です。

まず共用可能なデータパイプライン(**`dataset`**、**`dataloader`**)を作成し、最終的に**Pytorch**ベースの統一されたシンプルなアルゴリズムインターフェースを採用し、主要な汎用データセットでの精度アライメントを実行する計画です。

興味とコンピューティングリソースをお持ちの方の参加を歓迎し、一緒にYOLOモデルを手動で実装しましょう！！

### YOLO V3 from scratch

[YOLO V3 from scratch Notebook](./YOLOv3.ipynb) では使用するcoco8とcoco128データセットに対して**探索的データ分析(EDA)**を行っているので、必要に応じてご覧ください。

内容の概要は以下の通りです

- 手動YOLOv3実装
    - 主要参考コード
    - 実行環境と主要pythonライブラリ
    - データセットチェック
        - COCOYOLODataset
        - datasetとdataloaders
            - COCODataset
            - create_dataloaders
    - dataloaderテストと可視化
        - matplotlib日本語サポートの追加
        - plot_image_with_boxes
        - test_visualization
    - YOLOLoss
    - YOLOv3 Model
        - ConvLayer
        - ResBlock
        - make_conv_and_res_block
        - YoloLayer（DetectionBlockで使用）
        - DetectionBlock（Neckに似ている）
        - DarkNet53BackBone（Backbone）
        - YoloNetTail（Neck+Head）
        - YoloNetV3（Model）
        - test_yolov3_output_shapeテスト関数
    - YOLOEvaluator
        - low_confidence_suppression
        - non_max_suppression
        - calculate_map
        - calculate_ap
        - box_iou
        - evaluate_model
    - YOLOV3_Lightningと予測ボックスの可視化
        - YOLOV3_Lightning
        - visualize_predictions
        - test_nms_pipeline
        - training main関数
    - YOLOModule（上記コードの改善）
        - 問題
        - plot_training_metrics

### 本章参考資料

- [YOLOv3-Object-Detection-from-Scratch](https://github.com/williamcfrancis/YOLOv3-Object-Detection-from-Scratch/blob/main/YOLO_object_detection.ipynb)
- [YOLOv3-in-PyTorch](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py)
- [coco128 in kaggle](https://www.kaggle.com/datasets/ultralytics/coco128)
