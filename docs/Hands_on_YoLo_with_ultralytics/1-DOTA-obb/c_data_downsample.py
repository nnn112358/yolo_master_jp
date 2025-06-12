# -*- coding: utf-8 -*-
"""
# @file name  : c_data_downsample.py
# @author     : https://github.com/TingsongYu
# @date       : 2025年1月15日
# @brief      : データセットのダウンサンプリング、画像数の削減
"""

import os
import shutil
import random

BASE_DIR = os.path.dirname(__file__)
random.seed(42)

if __name__ == '__main__':

    data_root = r"G:\deep_learning_data\DOTAv1\DOTA-sub-split"
    new_data_root = data_root + '-downsample'
    if os.path.exists(new_data_root):
        shutil.rmtree(new_data_root)

    # サンプリング比率
    sampling_ratio = 0.1

    # train と val ディレクトリを処理
    for split in ['train', 'val']:
        # 元の画像とラベルパス
        split_image_dir = os.path.join(data_root, 'images', split)
        split_label_dir = os.path.join(data_root, 'labels', split)

        # 新しい画像とラベルパス
        new_split_image_dir = os.path.join(new_data_root, 'images', split)
        new_split_label_dir = os.path.join(new_data_root, 'labels', split)
        os.makedirs(new_split_image_dir, exist_ok=True)
        os.makedirs(new_split_label_dir, exist_ok=True)

        # 現在のsplitのすべての画像ファイルを取得
        label_files = [f for f in os.listdir(split_label_dir) if f.endswith(('.txt'))]
        random.shuffle(label_files)  # ランダムにシャッフル

        # サンプリングが必要な数量を計算
        sample_size = int(len(label_files) * sampling_ratio)
        sampled_files = label_files[:sample_size]  # 前10%を取得

        # サンプリング後の画像とラベルファイルを新しいディレクトリにコピー
        for label_file in sampled_files:
            # 画像をコピー
            image_file = label_file.replace('txt', 'jpg')

            original_image_path = os.path.join(split_image_dir, image_file)
            new_image_path = os.path.join(new_split_image_dir, image_file)
            shutil.copy(original_image_path, new_image_path)

            # 対応するラベルファイルをコピー
            original_label_path = os.path.join(split_label_dir, label_file)
            new_label_path = os.path.join(new_split_label_dir, label_file)

            if os.path.exists(original_label_path):
                shutil.copy(original_label_path, new_label_path)

        print(f"{split} ダウンサンプリング完了！合計 {len(sampled_files)} 枚の画像とラベルファイルをサンプリングしました。")