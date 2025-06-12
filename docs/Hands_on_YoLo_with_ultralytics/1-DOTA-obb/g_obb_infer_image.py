# -*- coding: utf-8 -*-
"""
# @file name  : g_obb_infer_image.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : モデル推論及び分析
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO
BASE_DIR = os.path.dirname(__file__)

if __name__ == '__main__':

    # weights_path = os.path.join(BASE_DIR, 'weights', 'yolo11m-obb.pt')  # モデル
    weights_path = os.path.join(BASE_DIR, 'runs', 'obb', 'train6', 'weights', 'best.pt')  # モデル
    image_path = r"G:\deep_learning_data\DOTAv1\DOTA-sub-split-downsample\images\val\P2231__640__590___1180.jpg"

    # ========================== step1: YOLOオブジェクトの初期化 ==========================
    model = YOLO(weights_path)

    # ========================== step2: 推論の実行 ==========================
    results = model(image_path, conf=0.1, iou=0.7)  # 推論設定
    annotated_frame = results[0].plot()  # 検出ボックス結果の描画

    # ========================== step3: OBB予測分析 ==========================
    xywhr_pred = np.array(results[0].obb[0].xywhr.to('cpu'))[0]
    print(xywhr_pred)

    # 回転矩形ボックスの4つの頂点を取得
    cx, cy, w, h, angle = xywhr_pred
    angle = angle * 180 / np.pi
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)  # 4つの頂点座標を取得
    box = np.int0(box)  # 整数に変換
    print("xywhr変換後に得られた4つの頂点：", box)

    # 画像上に回転矩形ボックスを描画
    cv2.polylines(annotated_frame, [box], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow("YOLO Inference", annotated_frame)
    cv2.waitKey(0)

