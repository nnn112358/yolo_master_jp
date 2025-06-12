# -*- coding: utf-8 -*-
"""
# @file name  : e_train.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/20
# @brief      : モデル比較実験、異なるモデルバージョン、モデルサイズ、画像入力サイズ、mosaic有効/無効、角度回転有効/無効を含む
"""

from ultralytics import YOLO
import itertools
import os
BASE_DIR = os.path.dirname(__file__)


def train_with_config(model_cfg, imgsz, data_aug_config, aug_id):
    # 実験名を生成
    model_name = model_cfg.replace("-obb.yaml", "")
    mosaic_status = "mosaic_on" if data_aug_config["mosaic"] == 1.0 else "mosaic_off"
    degrees_status = "degrees_180" if data_aug_config["degrees"] == 180 else "degrees_0"
    experiment_name = f"{model_name}_imgsz_{imgsz}_{mosaic_status}_{degrees_status}"  # 実験名

    print(f"Training with config: model={model_name}, imgsz={imgsz}, "
          f"mosaic={data_aug_config['mosaic']}, degrees={data_aug_config['degrees']}, "
          f"name={experiment_name}")

    # モデルを読み込み
    model = YOLO(model_cfg)

    # モデルを訓練
    results = model.train(
        data=dataset_path,  # データセット設定ファイル
        epochs=100,  # 訓練エポック数
        imgsz=imgsz,  # 入力解像度
        batch=4,  # バッチサイズ、90%のVRAMを使用
        mosaic=data_aug_config["mosaic"],  # Mosaicデータ拡張
        degrees=data_aug_config["degrees"],  # ランダム回転拡張
        name=experiment_name,  # 実験名
        amp=True,  # 混合精度訓練を無効化
    )

    return results


if __name__ == '__main__':

    # データセットパス
    dataset_path = os.path.join(BASE_DIR, 'cfg', 'DOTAv1-sub.yaml')  # データセット設定ファイル

    # モデル設定ファイルと入力解像度リストを定義
    versions = "v8 11".split()[::-1]  # 最大から開始してOOMを回避
    sizes = "n s m l".split()[::-1]
    model_cfgs = [f"yolo{version}{size}-obb.yaml" for version in versions for size in sizes]
    imgszs = [640, 960, 1280][::-1]

    # すべての組み合わせを生成
    experiment_configs = list(itertools.product(model_cfgs, imgszs))

    # データ拡張設定
    data_aug_configs = [
        {"mosaic": 0.0, "degrees": 0},  # 実験1：Mosaic無効、ランダム回転無効
        {"mosaic": 0.0, "degrees": 180},  # 実験2：Mosaic無効、ランダム回転有効
        {"mosaic": 1.0, "degrees": 0},  # 実験3：Mosaic有効、ランダム回転無効
        {"mosaic": 1.0, "degrees": 180},  # 実験4：Mosaic有効、ランダム回転有効
    ]

    # アブレーション実験を実行
    for model_cfg, imgsz in experiment_configs:
        for aug_id, data_aug_config in enumerate(data_aug_configs):
            train_with_config(model_cfg, imgsz, data_aug_config, aug_id)