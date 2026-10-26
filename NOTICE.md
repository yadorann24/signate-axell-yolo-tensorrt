# ライセンスと既存技術

このリポジトリのソースコードは [GNU Affero General Public License v3.0](LICENSE) で公開します。

YOLOモデルの定義とTensorRTへの変換には、外部依存として [Ultralytics 8.3.170](https://github.com/ultralytics/ultralytics/tree/v8.3.170) を利用します。同ライブラリはUltralyticsの著作物で、[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/v8.3.170/LICENSE) に基づいて提供されています。Ultralyticsのソースコード、配布重み、学習済み重みはこのリポジトリに同梱していません。

このプロジェクトで実装した部分は、変換条件の管理、入力検証、校正用画像の選択、実行ごとの出力管理、記録、CLI、負例のCopy-Pasteと学習処理への接続、テストです。YOLOのネットワーク構造、学習ループ、量子化アルゴリズムにはUltralyticsなどの外部実装を利用します。

PyTorch、ONNX、ONNX Runtime、NVIDIA TensorRTなどには各配布元のライセンスが適用されます。ソースコードのライセンスはコンペ配布画像の公開・再利用を許可するものではありません。配布画像は再配布やコンペ外利用を行わず、実行には利用権限のある入力を用意してください。
