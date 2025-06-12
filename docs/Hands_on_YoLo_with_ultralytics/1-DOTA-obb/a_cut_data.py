# -*- coding: utf-8 -*-
"""
# @file name  : a_cut_data.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : DOTAv1データセットに対してカテゴリクリッピングを実行し、訓練が必要なカテゴリを選択して、データセットを再生成します
"""

import os
import shutil


def cut_data(data_dir, out_dir, setname='train'):
    original_image_dir = os.path.join(data_dir, 'images', setname)
    original_label_dir = os.path.join(data_dir, 'labels', setname)
    new_image_dir = os.path.join(out_dir, 'images', setname)
    new_label_dir = os.path.join(out_dir, 'labels', setname)
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)

    counter = 0
    for label_file in os.listdir(original_label_dir):
        label_path = os.path.join(original_label_dir, label_file)
        new_label_path = os.path.join(new_label_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])

            # 指定されたカテゴリのターゲットのみを保持
            if class_id in keep_classes:
                # カテゴリインデックスを修正
                parts[0] = str(keep_classes[class_id])
                new_lines.append(' '.join(parts) + '\n')

        # 保持するターゲットがある場合、新しいラベルファイルと対応する画像を保存
        if new_lines:
            with open(new_label_path, 'w') as f:
                f.writelines(new_lines)

            image_file = label_file.replace('.txt', '.jpg')
            original_image_path = os.path.join(original_image_dir, image_file)
            new_image_path = os.path.join(new_image_dir, image_file)

            shutil.copy(original_image_path, new_image_path)
            counter += 1

    print("フィルタリングとマッピング完了！{}枚の画像を取得、保存先：{}".format(counter, out_dir))


if __name__ == '__main__':
    # 元データパス
    # ダウンロードリンク：https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip)
    data_dir = r"G:\deep_learning_data\DOTAv1"
    out_dir = os.path.join(data_dir, "DOTA-sub")  # 出力ディレクトリ

    # 保持する必要があるカテゴリとその新しいマッピング
    keep_classes_l = [0, 1]  # 選択されたカテゴリ、ここでは第0カテゴリと第1カテゴリを選択。
    keep_classes = {ori_class_idx: trg_index for trg_index, ori_class_idx in enumerate(keep_classes_l)}

    cut_data(data_dir, out_dir, setname='train')
    cut_data(data_dir, out_dir, setname='val')
