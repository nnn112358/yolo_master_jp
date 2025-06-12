# YoLo Master

![image](https://github.com/user-attachments/assets/f23752e3-e440-4fd6-a2ee-33b64bdc6544)

## Hands on YoLo with ultralytics

### プロジェクト構造
```
.
├── 0-dog-breed-detection
│   ├── README.md
│   ├── dog-breed-detection_kaggle.ipynb
│   ├── dog-breed-detection_local.ipynb
│   ├── runs
│   ├── test_dogs
│   └── yolov8m-v1
├── 1-DOTA-obb
│   ├── 0-tutorial_main.ipynb
│   ├── a_cut_data.py
│   ├── b_data_crop.py
│   ├── c_data_downsample.py
│   ├── d_data_visulization.py
│   ├── e_train.py
│   ├── f_eval.py
│   ├── g_obb_infer_image.py
│   ├── h_train_ablation.py
│   ├── i_logs_plot.py
│   ├── cfg
│   ├── runs
│   ├── README.md
│   └── utils_exp.py
└── README.md
```

### Feishuチュートリアル

- [0-dog-breed-detection YOLOシリーズ**入門実践チュートリアル**](https://wvet00aj34c.feishu.cn/docx/Ojcfd0ZF5olk4Yxwt9ZcjgSenUD?from=from_copylink)

    - 【Stanford Dogs Dataset basic tutorial with yoloV8】 本章ではdog-breed-detectionを通じて、ultralyticsを使用して犬種検出の実戦プロジェクトを素早く始める方法を示し、YOLOV8モデルの訓練および推論を実装し、学習者が実践を通じてyoloシリーズモデルの**入門操作**を習得できるよう支援します。
    - データセットは[**Stanford Dogs Dataset**](http://vision.stanford.edu/aditya86/ImageNetDogs/)由来


- [1-DOTA-obb YOLOシリーズ**上級実践チュートリアル**](https://wvet00aj34c.feishu.cn/docx/IPHFddAZmoBTr3xrRS0cW0Yanof?from=from_copylink)

    - 【DOTA-obb tutorial with yoloV11】 本章ではDOTAv1サンプルデータセットを通じて、ultralyticsプロジェクトで**YOLOV11-n OBBモデル**の訓練および推論を実装する方法を示し、学習者が実践を通じてyoloシリーズモデルの**上級操作**を習得できるよう支援します。
    - データセットは[**DOTA-v1.0**](https://captain-whu.github.io/DOTA/dataset.html)由来

### その他の実践

[YOLO V3 from scratch Notebook](../Pytorch_YoLo_From_Scratch/v3/YOLOv3_Hong.ipynb) では使用するcoco8とcoco128データセットに対して**探索的データ分析(EDA)**を行っているので、優先的にご覧ください。

### 本章参考資料

- [ultralytics](https://github.com/ultralytics/ultralytics)
- [ultralytics-tutorial](https://docs.ultralytics.com/tutorials/getting-started)
- [ultralytics-docs](https://docs.ultralytics.com/)
- [YOLOv3-Object-Detection-from-Scratch](https://github.com/williamcfrancis/YOLOv3-Object-Detection-from-Scratch/blob/main/YOLO_object_detection.ipynb)
- [YOLOv3-in-PyTorch](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py)




## 貢献に参加

- 問題を発見した場合、Issueでフィードバックを提供できます。返信がない場合は、[サポートチーム](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)のメンバーに連絡してフォローアップを求めてください〜
- 本プロジェクトに貢献したい場合、Pull requestを提供できます。返信がない場合は、[サポートチーム](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)のメンバーに連絡してフォローアップを求めてください〜
- Datawhaleに興味をお持ちで新しいプロジェクトを始めたい場合、[Datawhaleオープンソースプロジェクトガイド](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)に従って進めてください〜

## 私たちをフォロー

<div align=center>
<p>下のQRコードをスキャンして公式アカウントをフォロー：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="クリエイティブコモンズライセンス" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品は<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">クリエイティブコモンズ 表示-非営利-継承 4.0 国際ライセンス</a>の下でライセンスされています。

*注：デフォルトでCC 4.0ライセンスを使用し、プロジェクトの状況に応じて他のライセンスを選択することもできます*
