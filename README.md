# YOLO Master

## プロジェクト概要

- 本プロジェクトは主にYOLOシリーズモデルの紹介を行い、各バージョンモデルの構造、イノベーション、最適化、改良などを含みます
- 本コースの内容は、従来のDLコースにおいて、コンピュータビジョンモデルの中でも古典的なCVモデルであるResNetなどの後、TransformerなどのSequential Modelsの前の位置に大まかに位置します
- 本コースは学習者が主要なYOLOシリーズモデルの発展経緯を理解し、把握できるよう支援し、各自の応用分野においてさらなるイノベーションを図り、自身のタスクで良好な効果を達成することを目的としています。
- [**飛書ホワイトペーパー計画文書**](https://sxwqtaijh4.feishu.cn/docx/WNLJdo0wxoFPuExt6rbcvB8MnPg) / [**内部テスト文書**](https://wvet00aj34c.feishu.cn/docx/FwivdWGqMoYQPSxMotMcYVIrnOh)

## 対象読者

- 本コースは一定の**機械学習の基礎**を持ち、**Deep Learning**と**コンピュータグラフィックス**のコースを受講した学生、エンジニア、研究者を対象としています
- 応用領域はYOLOに基づく**物体検出**、**画像分類**、**画像セグメンテーション**、**姿勢検出**、**物体追跡**です（例：**[ultralytics実践](docs\Hands_on_YoLo_with_ultralytics)**）
- **YOLOアルゴリズムの手動実装（[From Scratch](docs\Pytorch_YoLo_From_Scratch)）**を期待する学習者、**YOLOシリーズモデルを自分の分野のデータに応用したり、性能を向上させたい（[Hacking](docs\Hacking_YoLo)）**エンジニア、研究者

## 目次

### 第一部 YOLO 全シリーズモデル詳解 ###

1. [YOLOv1详解](https://wvet00aj34c.feishu.cn/docx/U8STd5txXod1R5xhrrmcZh9fnTf) @刘伟鸿
2. [YOLOv2详解](https://wvet00aj34c.feishu.cn/docx/OHEhdwqXYoe8LIxwkRWcG0FLnnf) @蔡鋆捷
3. [YOLOv3详解](https://wvet00aj34c.feishu.cn/docx/U1e2dVfN3oFMUcxqkTWcNrNEnHr) @蔡鋆捷 @程宏
4. [YOLOv4详解](https://wvet00aj34c.feishu.cn/docx/IqGJdDvXsoNIGBxLsEWcGQGNnng) @蔡鋆捷 @胥佳程
5. [YOLOv5详解](https://wvet00aj34c.feishu.cn/docx/CltUdiVfMoaSkXxGaTvcpAyWnWh) @蔡鋆捷
6. [YOLOv6详解](https://wvet00aj34c.feishu.cn/docx/Clvbd8PDAoLD4Jx1Asdc6Afon0d) @陈国威
7. [YOLOv7详解](https://wvet00aj34c.feishu.cn/docx/K5eCdF7fSohwvfxVpeIcF0ZLnK9) @蔡鋆捷
8. [YOLOv8详解](https://ycnosmsebbdf.feishu.cn/docx/EqtRdOuy2oPnAkxkIE6cNhBsnwc) @蔡鋆捷 @程宏
9. [YOLOv9详解](https://sxwqtaijh4.feishu.cn/docx/FRJ6dPhALoqyC7xhVP6cwgSVn4e) @陈国威
10. [YOLOv10详解](https://wvet00aj34c.feishu.cn/docx/VagAdssMbo7a3exoagOcXr8BnAh) @陈国威 @李欣桓
11. [YOLO11详解](https://wvet00aj34c.feishu.cn/docx/ZUQ9d4LnmoYjv3xlBFTcprctnMg) @彭彩平
12. [YOLOX详解](https://wvet00aj34c.feishu.cn/docx/RCtddoe1joep4HxpmAPcYYBgnNc) @全政宇
13. [YOLOV12详解](https://wvet00aj34c.feishu.cn/docx/WrBydq19boEHN7xhp7pcLxd7n6f) @程宏 @张辉

### 第二部 YOLO 全シリーズチュートリアル ###

以下是日本語への翻訳です：

1. YOLO Master--YOLOを学ぶ正しい姿勢：入門から「真香」への奇妙な旅  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/U7YndF6zOo9Oi0xywBxcDvl7nNe)  @林涛 @程宏
2. YOLO系列モデルの鳥瞰図：YOLOv1-v11概要まとめとリリースタイムライン  [チュートリアル文書](https://sxwqtaijh4.feishu.cn/docx/Yc40ddMGIo7nOyxSXVZc6KztnYd) @程宏 @彭彩平 @張小白
3. YOLO系列アルゴリズムの基本原理とネットワーク構造  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/MKUhdQ9CmoIcR2x2TXrcBfY5ndh) @彭彩平 @程宏 @胡博毓
   1. YOLO系列モデルアルゴリズムにおけるLOSS [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/TGrYd5ttHonbzFxs1dgckacHnpb)  @谭斐然 @程宏 
   2. YOLO系列アルゴリズム原理のIoUまとめ  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/UUj2dE5aJoAMRixH9bIc7yxin7e)  @彭彩平 
   3. YOLO系列アルゴリズム原理の典型的ネットワークモジュール（詳細まとめ編）  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/MKUhdQ9CmoIcR2x2TXrcBfY5ndh)  @彭彩平  
4. YOLO系列アルゴリズム実践チュートリアル  @程宏 @余霆嵩 @刘伟鸿 @李欣桓 @谭斐然
   1. YOLO系列におけるultralyticsソースコードの読み方  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/K4d9d9B5KoaSPjxwOjXceBwKnih) 
   2. YOLO系列入門チュートリアル  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/Ojcfd0ZF5olk4Yxwt9ZcjgSenUD) / [チュートリアルコード](./Hands_on_YoLo_with_ultralytics/0-dog-breed-detection)
   3. YOLO系列アルゴリズム上級チュートリアル  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/U8STd5txXod1R5xhrrmcZh9fnTf) / [チュートリアルコード](./Hands_on_YoLo_with_ultralytics/1-DOTA-obb)
   4. YOLO実践における汎用データセット形式紹介と独自データセット作成 [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/Tdv4d2ZpmoWX4vxPPhfcvEIQnLh)
   5. YOLO実践におけるデータセット統合と自動ラベリング [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/R04QdmQMMoaA44xyDYkcA0AfnOd)
5. YOLO系列アルゴリズム改造チュートリアル  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/RXJKdo5ZJoT5QPxiV3vcpGPwnzX)  [チュートリアルコード](./Hacking_YoLo)  @白雪城 @謝彩承 @胡博毓
6. YOLO系列モデルのハードウェア展開と量子化  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/Oo71d5TjWoIzVPxaSIIc1Fysnqh)  @張小白 @白雪城 @程宏
7. YOLO Master ModelScope モデルアップロードチュートリアル  [チュートリアル文書](https://wvet00aj34c.feishu.cn/docx/VrZCdtOlvozI42xThc4cyxynnfg) @張小白 @程宏
8. YOLO系列アルゴリズムゼロから実装チュートリアル  [チュートリアル文書](./Pytorch_YoLo_From_Scratch) @刘伟鸿 @程宏 @蔡鋆捷 
   1. YOLOv1 [Notebook](./Pytorch_YoLo_From_Scratch/v1/YOLOv1.ipynb) / [README](./Pytorch_YoLo_From_Scratch/v1/README.md)
   2. YOLOv3 [Notebook](./Pytorch_YoLo_From_Scratch/v3/YOLOv3.ipynb) / [README](./Pytorch_YoLo_From_Scratch/v3/README.md)
   3. YOLOv5 [Notebook](./Pytorch_YoLo_From_Scratch/v5/YOLOv5.ipynb) / [README](./Pytorch_YoLo_From_Scratch/v5/README.md)
  
   4. 
### より多くのチュートリアルが完成・改善中（WIP）

コントリビューターの参加を歓迎し、一緒にチュートリアルを完善しましょう~
以下是日本語への翻訳です：

1. [YOLO系列アルゴリズムの基本原理とネットワーク構造 @彭彩平 @程宏](https://wvet00aj34c.feishu.cn/docx/MKUhdQ9CmoIcR2x2TXrcBfY5ndh)【完了】本文書は大から小（概要-->機能ブロック）、さらに小から大（重要概念-->典型的アルゴリズム-->典型的モジュール-->典型的ネットワーク構造）という紹介論理を採用

2. [YOLO系列モデルアルゴリズムにおけるLOSS](https://wvet00aj34c.feishu.cn/docx/TGrYd5ttHonbzFxs1dgckacHnpb) 【完了】最適化に使用されるLOSS関数のYOLO系列モデルアルゴリズムにおける応用と各バージョンでの異同と進化をまとめ @谭斐然

3. [YOLO系列モデルにおけるultralyticsソースコードの読み方](https://wvet00aj34c.feishu.cn/docx/K4d9d9B5KoaSPjxwOjXceBwKnih)【完了】ultralyticsソースコードの読み取りを試し、元モデルの修正を準備する学生のために特別に準備。より良いソースコード読解の支援を目的とし、実践チュートリアルの前に読むことを推奨 @谭斐然

4. [YOLO系列入門実践チュートリアル](https://wvet00aj34c.feishu.cn/docx/Ojcfd0ZF5olk4Yxwt9ZcjgSenUD)【ultralytics YOLOv8】【進行中】最適化とgithub notebook移植の調整 @北有青空

5. [YOLO系列モデルにおけるPP-YOLOEソースコードの読み方](https://wvet00aj34c.feishu.cn/docx/NvFwdZtD1owgx5xh6Enct1HcnLe)【進行中】国産深層学習フレームワークPaddlePaddleのアルゴリズムライブラリPaddleDetectionベース

6. [PP-YOLOE詳解](https://wvet00aj34c.feishu.cn/docx/F00CdJXU2ozAxixhCoLcu9v7nbh)【進行中】PP-YOLOEモデルの詳細解説

7. [PP-YOLOE系列モデル実践](https://wvet00aj34c.feishu.cn/docx/SxDodzBUlosqABxqvSJc606Cn4f)【進行中】PP-YOLOEモデルの実践

8. [mmyoloベースのYOLO系列アルゴリズム実践](https://wvet00aj34c.feishu.cn/docx/HM7LdVOHFolu07xkbovczWF5nRf)【進行中】mmyolo：OpenMMLab YOLO series toolbox and benchmark. RTMDet, RTMDet-Rotated,YOLOv5, YOLOv6, YOLOv7, YOLOv8,YOLOX, PPYOLOE等を実装

9. [YOLO実践における競技でのYOLO系列モデルの使用と最適化](https://wvet00aj34c.feishu.cn/docx/HHMrda1C5oRSCQxImGbcuIKlnGf) 【進行中】

10. [非YOLO系CVモデルの研究進展](https://wvet00aj34c.feishu.cn/docx/OWOMdjUHuoM60HxrdOrc4G0YnTg)【進行中】


## Github ディレクトリ構造説明

```text
.
├── docs
│   ├── Hacking_YoLo                      # カスタマイズチュートリアル
│   │   ├── C1 主干（Backbone）
│   │   ├── C2 颈部（Neck）
│   │   ├── C3 头部（Head）
│   │   ├── C4 注意力机制（Attention）
│   │   ├── C6 其他
│   │   └── README.md
│   ├── Hands_on_YoLo_with_ultralytics    # ultralyticsベースの応用実践チュートリアル
│   │   ├── 0-dog-breed-detection         # # 入門 YOLOv8m
│   │   ├── 1-DOTA-obb                    # # 上級OBBタスク YOLOv11n 
│   │   ├── 2-Beverages-Labeling          # # 高級データセット操作
│   │   └── README.md
│   ├── Images
│   └── Pytorch_YoLo_From_Scratch         # YOLOシリーズモデルゼロから実装チュートリアル
│       ├── datasets                      # # COCO demoデータセットを採用
│       │   ├── coco128.zip
│       │   └── coco8.zip
│       ├── README.md
│       ├── resource
│       ├── v1                            # # YOLOv1でVOCデータセットを使用
│       ├── v3                            # # YOLOv3でCOCO toyデータセットを使用
│       └── v5                            # # YOLOv5でCOCO toyデータセットを使用
└── README.md
```

## コントリビューターリスト

| 氏名 | 役割 | 紹介 |
| :-------| :---- | :---- |
| [程宏](https://github.com/chg0901) | プロジェクト主責任者、発起人、コードとチュートリアルの初回審査・内部テスト | DataWhale意向成员 |
| [蔡鋆捷](https://github.com/xinala-781) | プロジェクト主責任者、詳解コアコントリビューター、内部テスト組織 | DataWhale意向成员 |
| [余霆嵩](https://github.com/TingsongYu)| プロジェクト責任者、コード審査・最適化 | DataWhale意向成员 |
| [白雪城](https://github.com/JackBaixue) | プロジェクト責任者、発起人、カスタマイズ責任者 | DataWhale成员 |
| [彭彩平](https://github.com/caipingpeng) | プロジェクト責任者、YOLO俯瞰、基本原理とネットワーク構造 | |
| [刘伟鸿](https://github.com/Weihong-Liu) | V1詳解、V1 Scratch、データセット作成 | DataWhale成员 |
| [胡博毓](https://github.com/HuBoyu021124) | V8 Review、カスタマイズチュートリアル | DataWhale成员 |
| [谢彩承](https://github.com/YoungBossX) | V1、V2、V5 Review、カスタマイズチュートリアル |DataWhale意向成员 |
| [陈国威](https://github.com/gomevie) | V6、V9、V10詳解 |DataWhale意向成员  |
| [全政宇](https://github.com/EdQinHUST) |V9、V10 Review、YOLOX詳解| DataWhale意向成员  |
| [张小白](https://www.zhihu.com/people/zhanghui_china) | YOLOシリーズモデルのトリビアと噂、ハードウェア展開、ModelScope使用|DataWhale意向成员  |
| [李欣桓](https://github.com/NorthBlueSky) |**V10**、V11 Review，データセット統合とラベリング | 安徽理工大学  |
| [胥佳程](https://github.com/Thedan-1) | **V4**、V5 Review | DataWhale意向成员、青岛科技大学|
| 徐韵婉 | 発起人、飛書チュートリアル管理・メンテナンス | DataWhale成员 |

注：プロジェクト責任者の実際の貢献内容はリストにすべて表示することはできません。各責任者の努力と継続的なフォローアップに感謝いたします。

### Reviewer List

YOLO Masterプロジェクトの開発と最適化プロセスにおいて、各レビューアーは厳密な専門性、細かい審査意見、建設的なフィードバックで、
プロジェクト品質の向上に重要な支援と貴重な提案を提供し、チュートリアルプロジェクトの知識体系の完全性と合理性を著しく向上させ、学習時のユーザーエクスペリエンスを改善しました。
皆様の専門的洞察と忍耐強い指導はYOLO Masterプロジェクトの継続的な反復の基礎であり、将来の開発とチュートリアルシステムの完全化プロセスにおいて皆様との継続的な協力を期待しています。
**YOLOMaster**は皆様と共に成長し進歩し、共にプロジェクトを**より専門的で、より効率的で、より使いやすいYOLOシリーズモデルのオープンソース学習チュートリアル**にするために努力しましょう！

ここで、**第一次内部テスト**（[**文書**](https://wvet00aj34c.feishu.cn/docx/FwivdWGqMoYQPSxMotMcYVIrnOh)）に参加されたレビューアーの皆様、
すでにチュートリアルとコードを貢献し、コントリビューターになった仲間たち（*斜体*）、
さらに多くの仲間が内部テストに参加し、より多くの仲間が私たちのプロジェクトチュートリアルのコントリビューターになってくださることを期待しています！

|Ver. No | Reviewer Name(s) |
| :-------| :-------- |
|v1|*谢彩承*|
|v2|*谢彩承* |
|v3|[冯启洪](https://github.com/fqhyyds) (汕头大学)，*[谭斐然](https://github.com/frtanxidian)(DataWhale意向成员、西安电子科技大学)* |
|v4|*胥佳程* |
|v5|*谢彩承*，*胥佳程* |
|v6| 马恺 |
|v7|*[林涛](www.lintao.online)(DataWhale意向成员)* |
|v8|揭沁沅，[冯启洪](https://github.com/fqhyyds) (汕头大学)，*胡博毓* |
|v9|*全政宇* |
|v10|*全政宇*，*李欣桓* |
|v11|[冯启洪](https://github.com/fqhyyds) (汕头大学)，*李欣桓* |

注：斜体の人員はプロジェクトコントリビューターです。より多くのレビューアーがコントリビューターになってくださることを期待しています！

## 貢献方法

- 問題を発見した場合は、Issueでフィードバックしてください。返信がない場合は[サポートチーム](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)に連絡してフォローアップを受けてください~
- 本プロジェクトに貢献したい場合は、Pull requestを提出してください。返信がない場合は[サポートチーム](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)に連絡してフォローアップを受けてください~
- Datawhaleに興味を持ち、新しいプロジェクトを開始したい場合は、[Datawhaleオープンソースプロジェクトガイド](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)に従って操作してください~

## フォローしてください

<div align=center>
<p>下のQRコードをスキャンして公式アカウントをフォロー：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品は<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">クリエイティブ・コモンズ 表示-非営利-継承 4.0 国際ライセンス</a>の下でライセンスされています。

*注：デフォルトでCC 4.0ライセンスを使用していますが、プロジェクトの状況によって他のライセンスを選択することもできます*
