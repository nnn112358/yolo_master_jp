# -*- coding: utf-8 -*-
"""
# @file name  : utils_exp.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : DOTA モデル訓練ツール関数ライブラリ
"""
import os
import glob
import shutil
import math
import os
import random
import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def cut_data(data_dir, out_dir, keep_classes, setname='train'):
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


def count_classes_in_file(label_path):
    """
    単一のラベルファイル内のカテゴリを統計
    :param label_path: ラベルファイルパス
    :return: カテゴリIDを含むリスト
    """
    with open(label_path, "r") as f:
        lines = f.readlines()
    return [int(line.split()[0]) for line in lines if line.strip()]


def count_classes_in_files(label_files):
    """
    複数のラベルファイル内のカテゴリを統計
    :param label_files: ラベルファイルパスのリスト
    :return: すべてのカテゴリIDを含むリスト
    """
    class_ids = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(count_classes_in_file, label_files)
        for result in results:
            class_ids.extend(result)
    return class_ids


def print_yolo_dataset_info(dataset_path):
    """
    YOLOデータセットの基本情報を出力
    :param dataset_path: YOLOデータセットのルートディレクトリパス
    """
    # パスを定義
    images_train_path = os.path.join(dataset_path, "images", "train")
    images_val_path = os.path.join(dataset_path, "images", "val")
    labels_train_path = os.path.join(dataset_path, "labels", "train")
    labels_val_path = os.path.join(dataset_path, "labels", "val")

    # パスが存在するかチェック
    if not all(
            os.path.exists(path) for path in [images_train_path, images_val_path, labels_train_path, labels_val_path]):
        print("エラー：データセットディレクトリ構造がYOLO形式に準拠していません！")
        print([images_train_path, images_val_path, labels_train_path, labels_val_path])
        return

    # 画像数を統計
    train_images = glob.glob(os.path.join(images_train_path, "*"))
    val_images = glob.glob(os.path.join(images_val_path, "*"))
    total_images = len(train_images) + len(val_images)

    # ラベル数を統計
    train_labels = glob.glob(os.path.join(labels_train_path, "*"))
    val_labels = glob.glob(os.path.join(labels_val_path, "*"))
    total_labels = len(train_labels) + len(val_labels)

    # カテゴリ数を統計
    all_label_files = train_labels + val_labels
    class_ids = count_classes_in_files(all_label_files)
    class_counts = defaultdict(int)
    for class_id in class_ids:
        class_counts[class_id] += 1

    # 基本情報を出力
    print(f"データセット{dataset_path}の基本情報：")
    print(f"訓練セット画像数: {len(train_images)}")
    print(f"検証セット画像数: {len(val_images)}")
    print(f"総画像数: {total_images}")
    print(f"訓練セットラベル数: {len(train_labels)}")
    print(f"検証セットラベル数: {len(val_labels)}")
    print(f"総ラベル数: {total_labels}")
    print(f"カテゴリ数: {len(class_counts)}")
    print("カテゴリ分布：")
    for class_id, count in sorted(class_counts.items()):
        print(f"  カテゴリ {class_id}: {count} 個のインスタンス")


def parse_annotation(annotation_path):
    """
    YOLO形式のアノテーションファイルを解析し、アノテーション情報を返す
    """
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 9:  # YOLO形式：class_id x1 y1 x2 y2 x3 y3 x4 y4
                class_id = int(parts[0])  # カテゴリID
                coords = list(map(float, parts[1:]))  # 正規化されたOBB頂点座標を抽出
                annotations.append((class_id, coords))
    return annotations


def draw_obb(image, annotations, img_width, img_height):
    """
    画像上に有向境界ボックス（OBB）とカテゴリラベルを描画
    """
    for ann in annotations:
        class_id, coords = ann
        # 正規化座標を実際のピクセル座標に変換
        points = [(int(coords[i] * img_width), int(coords[i + 1] * img_height)) for i in range(0, 8, 2)]
        # OBBを描画
        for i in range(4):
            cv2.line(image, points[i], points[(i + 1) % 4], (0, 255, 0), 2)
        # カテゴリラベルを描画
        cv2.putText(image, str(class_id), points[0], cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    return image


def visualize_random_images(yolo_data_dir, setname='train', num_images=4):
    """
    データセット内の画像をランダムに選択して可視化
    """
    images_dir = os.path.join(yolo_data_dir, 'images', setname)  # 画像パス
    labels_dir = os.path.join(yolo_data_dir, 'labels', setname)  # ラベルパス

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

        # OBBとカテゴリラベルを描画
        image_with_boxes = draw_obb(image, annotations, img_width, img_height)

        # 画像を表示
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image_with_boxes)
        plt.title(image_file)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def check_yolo_yaml_paths(yaml_path):
    """
    YOLO YAMLファイル内のパスが存在するかチェック
    :param yaml_path: YAMLファイルのパス
    """
    # YAMLファイルが存在するかチェック
    if not os.path.exists(yaml_path):
        print(f"エラー：YAMLファイル '{yaml_path}' が存在しません！")
        return

    # YAMLファイルを読み込み（エンコーディングをutf-8に指定）
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"エラー：YAMLファイル '{yaml_path}' を解析できません、理由：{e}")
            return
        except UnicodeDecodeError as e:
            print(f"エラー：ファイルエンコーディングがutf-8ではありません、ファイル '{yaml_path}' のエンコーディング形式を確認してください。")
            return

    # 訓練、検証パスが存在するかチェック
    root_dir = yaml_data['path']
    for key in ["train", "val"]:
        if key not in yaml_data:
            print(f"警告：YAMLファイルにフィールド '{key}' がありません")
            continue

        paths = yaml_data[key]
        if paths is None:
            print(f"警告：フィールド '{key}' の値がNoneです")
            continue

        # パスが文字列の場合、リストに変換
        if isinstance(paths, str):
            paths = [paths]

        # 各パスが存在するかチェック
        for path in paths:
            abs_path = os.path.abspath(os.path.join(root_dir, path))  # 絶対パスに変換
            if not os.path.exists(abs_path):
                print(f"警告!：パス '{abs_path}' が存在しません!")
                print(f"警告!：パス '{abs_path}' が存在しません!!")
                print(f"警告!：パス '{abs_path}' が存在しません!!!")


def visualize_yolo_logs(log_dir, setname='train'):
    """
    YOLO訓練ログ内の画像ファイルを可視化
    :param log_dir: ログディレクトリパス（例：'runs/train/exp'）
    """
    # ログディレクトリが存在するかチェック
    if not os.path.exists(log_dir):
        print(f"エラー：ログディレクトリ '{log_dir}' が存在しません！")
        return

    # 可視化が必要な画像ファイル名を定義
    log_images = {
        "results.png": "results.png",
        "PR_curve.png": "PR_curve.png",
        "confusion_matrix.png": "confusion_matrix.png"
    }
    if setname == 'val':
        del log_images["results.png"]

    # すべての画像を描画するためのfigureを作成
    fig, axes = plt.subplots(1, len(log_images), figsize=(24, 8))
    fig.suptitle("YOLO logs", fontsize=16)

    # 画像ファイルを走査して描画
    for i, (image_name, title) in enumerate(log_images.items()):
        image_path = os.path.join(log_dir, image_name)
        if not os.path.exists(image_path):
            print(f"警告：画像ファイル '{image_name}' が存在しません！")
            continue

        # 画像を読み込み
        img = mpimg.imread(image_path)

        # 画像を描画
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")  # 座標軸をオフ

    # レイアウトを調整して表示
    # plt.tight_layout()
    plt.show()