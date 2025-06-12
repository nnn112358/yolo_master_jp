# -*- coding: utf-8 -*-
"""
# @file name  : b_data_crop.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : 画像のタイル分割処理
"""

import os
from ultralytics.data.split_dota import split_test, split_trainval

BASE_DIR = os.path.dirname(__file__)

if __name__ == '__main__':

    data_dir = r"G:\deep_learning_data\DOTAv1\DOTA-sub"  # 前のステップで取得したデータディレクトリ
    out_dir = os.path.join(os.path.dirname(data_dir), 'DOTA-sub-split')

    # データセット画像のタイル分割処理により、画像解像度を「下げる」
    split_trainval(data_root=data_dir, save_dir=out_dir, rates=[1.0], crop_size=640, gap=50)  # gapは重複するピクセルを表す
