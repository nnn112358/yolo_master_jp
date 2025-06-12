# -*- coding: utf-8 -*-
"""
# @file name  : e_train.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : モデル訓練
"""

import os
from ultralytics import YOLO
BASE_DIR = os.path.dirname(__file__)

"""
ultralyticsの回転ターゲット検出コードの問題により、訓練中の検証コード内のNMSがGPUメモリの急激な増加を引き起こす可能性があります。
現在、この問題に対する適切な解決策はありませんが、max_nms数を修正して回避することをお勧めします。
具体的な操作：
まずultralyticsのインストールディレクトリを見つけ、次に対応する.pyファイル：ultralytics/models/yolo/obb/val.pyを見つけます。
ops.non_max_suppression関数に、新しい入力パラメータとしてmax_nms=1920を追加します。
"""
import ultralytics
install_dir = os.path.dirname(ultralytics.__file__)
val_path = os.path.join(install_dir, "models", 'yolo', 'obb', 'val.py')
print(f"ultralytics インストールディレクトリ: {install_dir} \n修正が必要なファイルパス:{val_path}")

if __name__ == '__main__':

    dataset_path = os.path.join(BASE_DIR, 'cfg', 'DOTAv1-sub.yaml')  # データセット
    # weights_path = os.path.join(BASE_DIR, 'weights', 'yolo11n-obb.pt')  # モデル
    yolo11n_obb_path = os.path.join(BASE_DIR, 'cfg', 'yolo11s-obb.yaml')  # モデル

    # Load a model
    # model = YOLO(weights_path)
    model = YOLO(yolo11n_obb_path)

    # Train the model
    results = model.train(data=dataset_path, epochs=100, amp=False, batch=8)

