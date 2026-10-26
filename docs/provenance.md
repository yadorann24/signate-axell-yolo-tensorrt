# 使用ライブラリと出典

このリポジトリはAxell AI Contest 2025への参加に関連して作成した、対象外物体のCopy-PasteとYOLOのTensorRT変換の実装です。モデル生成、データ拡張と学習への接続、変換設定、入力検証、実行ディレクトリの管理、推論計測を提供します。

Copy-Pasteは、明示した素材クラスの矩形だけを切り出し、正解枠を避けて配置します。学習への接続はUltralyticsのデータセットとepochイベントを利用します。配布データ、貼り付け素材、学習済みモデルは同梱せず、利用者が権利を確認した入力を指定します。

## 実装の役割分担

| 部分 | 使用する実装 |
| --- | --- |
| モデル名・presetの管理、CLI、入力と出力の検証、学習画像を読むcalibration loader | 本リポジトリのコード |
| 負例素材の選別と切り出し、重なりを避ける配置、学習データへの接続と終盤の無効化 | 本リポジトリのコード |
| YOLOのモデル構成、checkpointの読み込み、exportの呼び出し | Ultralytics 8.3.170 |
| 標準の学習用拡張、データローダー、損失計算、最適化、検証とcheckpoint保存 | Ultralytics 8.3.170、PyTorch |
| テンソルとニューラルネットワーク | PyTorch |
| ONNXへの変換とグラフの簡略化 | ONNX、onnxslim、およびUltralyticsのexport処理 |
| GPU向けengineの構築とINT8 calibration | NVIDIA TensorRT、およびUltralyticsのexport処理 |
| 素材一覧とcalibration設定の読み込み | PyYAML |
| 素材・画像の読込とリサイズ、矩形の画素操作 | OpenCV、NumPy |
| calibrationのLetterBoxによる余白付きのサイズ調整 | Ultralytics |

本リポジトリの工夫は、学習用の負例を選別して正解ラベルを保ちながら追加すること、workerを使う学習処理へ拡張を確実に接続すること、変換の構成・入力範囲・出力・実行結果を明示して確認しやすくすることです。モデルのネットワーク、損失関数、最適化、TensorRTの量子化には上記ライブラリの実装を使用しています。

## 技術資料

- [Ultralytics 8.3.170](https://github.com/ultralytics/ultralytics/tree/v8.3.170)：使用バージョン。
- [YOLO11の構成](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/cfg/models/11/yolo11.yaml)：モデル規模に応じた構成。
- [YOLO12の構成](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/cfg/models/12/yolo12.yaml)：モデル規模に応じた構成。
- [学習ループ](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/engine/trainer.py)：学習・検証とepochイベント。
- [データセット](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/data/dataset.py)、[データローダー](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/data/build.py)：変換の構築、Mosaic終了とworkerの管理。
- [学習用拡張とFormat](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/data/augment.py)：画像・正解枠の変換とテンソル化。
- [exportとcalibrationの実装](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/engine/exporter.py)：export引数、ONNX変換、calibration対象の選択。
- [TensorRT engineの生成処理](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/utils/export.py)：数値精度、calibration cache、engineの保存。
- [PyTorch 2.4.0の導入方法](https://pytorch.org/get-started/previous-versions/#v240)：CUDA対応版の配布元。

挙動の説明は使用する固定バージョンの実装に合わせています。ライブラリの最新版では、依存関係や変換時の既定値が変わる場合があります。

## コンペティションの資料

- [Axell AI Contest 2025 公式競技ページ](https://user.competition.signate.jp/ja/competition/detail/?competition=6e644a0277314ba2a1d36a1634f75f41&task=cc969d0e55a44990becf0c0688235765)：課題、評価条件、公開条件。
- [主催者による開催告知](https://www.axell.co.jp/news/20250714.html)：飲料銘柄を判定するモデル開発という課題の背景。
- [公式参加規約](https://api.competition.signate.jp/api/competition/6e644a0277314ba2a1d36a1634f75f41/term)：公開時の条件と特記事項。

## ライセンス

本リポジトリのコードは[AGPL-3.0](../LICENSE)で公開します。

Ultralytics 8.3.170は[GNU AGPLのライセンス本文](https://github.com/ultralytics/ultralytics/blob/v8.3.170/LICENSE)を同梱しています。利用条件は[Ultralyticsの公式ライセンス説明](https://www.ultralytics.com/license)も参照してください。その他の依存ライブラリにも、それぞれのライセンスが適用されます。

コードのライセンスは、SIGNATE配布画像やcheckpointの利用・再配布を許可するものではありません。配布画像は再配布やコンペ外利用を行わず、実行には利用権限のある入力を用意してください。
