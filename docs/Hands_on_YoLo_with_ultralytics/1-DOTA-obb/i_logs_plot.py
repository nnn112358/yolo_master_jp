# -*- coding: utf-8 -*-
"""
# @file name  : i_logs_plot.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/20
# @brief      : 対比実験ログの可視化
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
BASE_DIR = os.path.dirname(__file__)


def plot_log_model_size(log_dir):
    # モデルバージョンと対応する色を定義
    models = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    # データを格納する辞書を初期化
    data = {model: {'imgsz': [], 'map50_95': []} for model in models}

    # ログフォルダを走査
    for folder in os.listdir(log_dir):
        #
        re_1 = r'(yolo11|yolov8)([nslm])_imgsz_(\d+)_mosaic_(on)_degrees_(0)'
        # 正規表現を使用してフォルダ名を解析
        match = re.match(re_1, folder)
        if match:
            model_type = match.group(1) + match.group(2)  # 例：'yolo11n'
            imgsz = int(match.group(3))  # 画像サイズ
            mosaic = match.group(4)  # mosaicスイッチ
            degrees = int(match.group(5))  # 回転角度

            # results.csvファイルを読み取り
            csv_path = os.path.join(log_dir, folder, 'results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # metrics/mAP50-95(B)列の最大値を取得
                max_map = df['metrics/mAP50-95(B)'].max()
                # データを辞書に格納
                data[model_type]['imgsz'].append(imgsz)
                data[model_type]['map50_95'].append(max_map)

    # 曲線を描画
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        if data[model]['imgsz']:  # データがあることを確認
            # 画像サイズで並び替え
            sorted_imgsz, sorted_map = zip(*sorted(zip(data[model]['imgsz'], data[model]['map50_95'])))
            # 線種を設定：yolo11は実線、yolov8は破線
            linestyle = '-' if model.startswith('yolo11') else '--'
            plt.plot(sorted_imgsz, sorted_map, label=model, color=color, linestyle=linestyle, marker='o')

    # グラフのタイトルとラベルを設定
    plt.title('mAP50-95 vs Image Size')
    plt.xlabel('Image Size')
    plt.ylabel('mAP50-95')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mosaic(log_dir, model_version='yolov8'):
    # モデルバージョンと対応する色を定義
    model_sizes = "n s m l".split()
    models = [model_version + size for size in model_sizes]
    colors = ['b', 'g', 'r', 'c']  # モデルバージョンを区別するための色
    line_styles = ['-', '--']  # 実線はmosaic有効、破線はmosaic無効を表す

    # データを格納する辞書を初期化
    data = {model: {'imgsz_mosaic_on': [], 'map50_95_mosaic_on': [], 'imgsz_mosaic_off': [], 'map50_95_mosaic_off': []} for
            model in models}

    # ログフォルダを走査
    for folder in os.listdir(log_dir):
        # 正規表現を使用してフォルダ名を解析
        pattern = r'({})([nslm])_imgsz_(\d+)_mosaic_(on|off)_degrees_(0)'.format(re.escape(model_version))
        match = re.match(pattern, folder)
        # match = re.match(r'(model_version)([nslm])_imgsz_(\d+)_mosaic_(on|off)_degrees_(0)', folder)
        if match:
            model_type = match.group(1) + match.group(2)  # 例：'yolov8n'
            imgsz = int(match.group(3))  # 画像サイズ
            mosaic = match.group(4)  # mosaicスイッチ状態
            degrees = int(match.group(5))  # 回転角度

            # results.csvファイルを読み取り
            csv_path = os.path.join(log_dir, folder, 'results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # metrics/mAP50-95(B)列の最大値を取得
                max_map = df['metrics/mAP50-95(B)'].max()
                # mosaic状態に応じてデータを格納
                if mosaic == 'on':
                    data[model_type]['imgsz_mosaic_on'].append(imgsz)
                    data[model_type]['map50_95_mosaic_on'].append(max_map)
                else:
                    data[model_type]['imgsz_mosaic_off'].append(imgsz)
                    data[model_type]['map50_95_mosaic_off'].append(max_map)

    # 曲線を描画
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        # mosaic有効の曲線を描画
        if data[model]['imgsz_mosaic_on']:
            sorted_imgsz, sorted_map = zip(*sorted(zip(data[model]['imgsz_mosaic_on'], data[model]['map50_95_mosaic_on'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (mosaic on)', color=color, linestyle=line_styles[0],
                     marker='o')
        # mosaic無効の曲線を描画
        if data[model]['imgsz_mosaic_off']:
            sorted_imgsz, sorted_map = zip(
                *sorted(zip(data[model]['imgsz_mosaic_off'], data[model]['map50_95_mosaic_off'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (mosaic off)', color=color, linestyle=line_styles[1],
                     marker='x')

    # グラフのタイトルとラベルを設定
    plt.title(f'mAP50-95 vs Image Size ({model_version}with Mosaic On/Off)')
    plt.xlabel('Image Size')
    plt.ylabel('mAP50-95')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_degrees(log_dir, model_version='yolov8'):
    # モデルバージョンと対応する色を定義
    model_sizes = "n s m l".split()
    models = [model_version + size for size in model_sizes]
    # モデルバージョンと対応する色を定義
    colors = ['b', 'g', 'r', 'c']  # モデルバージョンを区別するための色
    line_styles = ['-', '--']  # 実線はdegrees=0、破線はdegrees=180を表す

    # データを格納する辞書を初期化
    data = {
        model: {'imgsz_degrees_0': [], 'map50_95_degrees_0': [], 'imgsz_degrees_180': [], 'map50_95_degrees_180': []}
        for model in models}

    # ログフォルダを走査
    for folder in os.listdir(log_dir):
        # 正規表現を使用してフォルダ名を解析
        pattern = r'({})([nslm])_imgsz_(\d+)_mosaic_(on)_degrees_(\d+)'.format(re.escape(model_version))
        match = re.match(pattern, folder)
        # match = re.match(r'(yolov8)([nslm])_imgsz_(\d+)_mosaic_(on)_degrees_(\d+)', folder)
        if match:
            model_type = match.group(1) + match.group(2)  # 例：'yolov8n'
            imgsz = int(match.group(3))  # 画像サイズ
            mosaic = match.group(4)  # mosaicスイッチ状態
            degrees = int(match.group(5))  # 回転角度

            # results.csvファイルを読み取り
            csv_path = os.path.join(log_dir, folder, 'results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # metrics/mAP50-95(B)列の最大値を取得
                max_map = df['metrics/mAP50-95(B)'].max()
                # degrees状態に応じてデータを格納
                if degrees == 0:
                    print(folder)
                    data[model_type]['imgsz_degrees_0'].append(imgsz)
                    data[model_type]['map50_95_degrees_0'].append(max_map)
                elif degrees == 180:
                    print(folder)
                    data[model_type]['imgsz_degrees_180'].append(imgsz)
                    data[model_type]['map50_95_degrees_180'].append(max_map)

    # 曲線を描画
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        # degrees=0の曲線を描画
        if data[model]['imgsz_degrees_0']:
            sorted_imgsz, sorted_map = zip(
                *sorted(zip(data[model]['imgsz_degrees_0'], data[model]['map50_95_degrees_0'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (degrees=0)', color=color, linestyle=line_styles[0],
                     marker='o')
        # degrees=180の曲線を描画
        if data[model]['imgsz_degrees_180']:
            sorted_imgsz, sorted_map = zip(
                *sorted(zip(data[model]['imgsz_degrees_180'], data[model]['map50_95_degrees_180'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (degrees=180)', color=color, linestyle=line_styles[1],
                     marker='x')

    # グラフのタイトルとラベルを設定
    plt.title(f'mAP50-95 vs Image Size ({model_version} with Degrees=0 vs Degrees=180)')
    plt.xlabel('Image Size')
    plt.ylabel('mAP50-95')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    # フォルダパスを定義
    log_dir = os.path.join(BASE_DIR, 'runs', 'csv_files', 'obb')

    # 実験一：異なる画像サイズ、異なるモデルバージョン間の差異を観察
    plot_log_model_size(log_dir)

    # 実験二：mosaicデータ拡張の有効・無効の差異を観察
    plot_mosaic(log_dir, 'yolov8')

    # 実験三：回転角度の有効・無効の差異を観察
    plot_degrees(log_dir, 'yolov8')