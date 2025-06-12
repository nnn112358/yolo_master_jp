# -*- coding: utf-8 -*-
"""
# @file name  : d_data_visulization.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : DOTAv1 データセットラベルの可視化
"""
import math
import os
import random
import cv2
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(__file__)


def parse_annotation(annotation_path):
    """
    YOLO形式のアノテーションファイルを解析し、アノテーション情報を返す
    """
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 9:  # YOLO形式：class_id x1 y1 x2 y2 x3 y3 x4 y4
                class_id = int(parts[0])  # クラスID
                coords = list(map(float, parts[1:]))  # 正規化されたOBB頂点座標を抽出
                annotations.append((class_id, coords))
    return annotations


def draw_obb(image, annotations, img_width, img_height):
    """
    画像上に有向境界ボックス（OBB）とクラスラベルを描画
    """
    for ann in annotations:
        class_id, coords = ann
        # 正規化座標を実際のピクセル座標に変換
        points = [(int(coords[i] * img_width), int(coords[i + 1] * img_height)) for i in range(0, 8, 2)]
        # OBBを描画
        for i in range(4):
            cv2.line(image, points[i], points[(i + 1) % 4], (0, 255, 0), 2)
        # クラスラベルを描画
        cv2.putText(image, str(class_id), points[0], cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    return image


def visualize_random_images(images_dir, labels_dir, num_images=4):
    """
    データセット内の画像をランダムに選択して可視化
    """
    # すべての画像ファイル名を取得
    labels_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    selected_images = random.sample(labels_files, min(num_images, len(labels_files)))  # ランダムに画像を選択

    # subplot レイアウトを動的に調整
    num_rows = math.ceil(math.sqrt(num_images))  # 行数
    num_cols = math.ceil(num_images / num_rows)  # 列数

    # キャンバスを作成
    plt.figure(figsize=(num_cols * 5, num_rows * 5))  # 画像数に応じてキャンバスサイズを調整

    # 選択された画像を可視化
    for i, label_file in enumerate(selected_images):
        # 画像を読み込み
        image_file = label_file.replace('.txt', '.jpg')  # ラベルファイル名と画像ファイル名は一致
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB形式に変換
        img_height, img_width, _ = image.shape  # 画像の幅と高さを取得

        # アノテーションを読み込み
        annotation_path = os.path.join(labels_dir, label_file)
        annotations = parse_annotation(annotation_path)

        # OBBとクラスラベルを描画
        image_with_boxes = draw_obb(image, annotations, img_width, img_height)

        # 画像を表示
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image_with_boxes)
        plt.title(image_file)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # データセットパスを設定
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip

    images_dir = r'G:\deep_learning_data\DOTAv1\DOTA-sub-split\images\train'  # 画像パス
    labels_dir = r'G:\deep_learning_data\DOTAv1\DOTA-sub-split\labels\train'  # ラベルパス

    # ランダムにN枚の画像を可視化
    visualize_random_images(images_dir, labels_dir, num_images=4)