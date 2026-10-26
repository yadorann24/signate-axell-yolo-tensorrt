# 推論・ベンチマークと比較実験

`axell-benchmark`は、自分で利用権限を持つ画像とローカルの `.pt` / `.engine` を使い、同じ条件で推論時間・容量・出力を記録するコマンドです。競技の配布画像をコンペ外の計測や動作例に使用しないでください。以下で保存する画像パス・予測・実行記録も非公開で管理します。

## 入力と環境

PyTorchの推論は[README](../README.md)のruntime環境で実行できます。engineにはGPU用export環境が必要です。`pip install --no-deps -e .` でパッケージを登録すると、`axell-benchmark` が利用できます。対応はbatch=1、CPUまたはCUDA device 0です。checkpointは利用権限と出所が分かるものを使用します。

画像リストは1行1パスのUTF-8テキストです。相対パスはリストのあるディレクトリから解決し、同じファイルの重複やURLを拒否します。実行時に対象画像の内容ハッシュを記録します。

```text
owned-images/validation/scene_a.png
owned-images/validation/scene_b.png
```

独立した検証画像を使い、INT8 calibrationには学習側の画像だけを渡します。ベンチマークの画像一覧をcalibrationへ転用しないでください。

```bash
axell-benchmark \
  --weights /path/private/best.pt \
  --images /path/private/evaluation-images.txt \
  --output-dir /path/private/benchmarks/pytorch-fp32 \
  --image-size 512 768 --device 0 --warmup 10 --repeats 10 \
  --conf 0.001 --iou 0.7 --max-det 5
```

入力サイズは高さ・幅の順で、32の正の倍数を指定します。`rect=False` により、全バックエンドで指定サイズのLetterBoxを使います。RGB変換・255除算・NMS・元画像への座標復元は固定Ultralytics版のpredictorで行います。PyTorchをFP16で測る場合は `--half` を追加します。CPUでは使えません。

## TensorRT engineと比較する

engineには、対応する `export.json` を必須としています。engineのSHA-256、成功状態、入力サイズ・batchの対応を確認し、元checkpointのSHA-256と変換条件も結果に残します。

```bash
axell-benchmark \
  --weights /path/private/runs/int8-512x768/model.engine \
  --export-manifest /path/private/runs/int8-512x768/export.json \
  --images /path/private/evaluation-images.txt \
  --output-dir /path/private/benchmarks/tensorrt-int8 \
  --image-size 512 768 --device 0 --warmup 10 --repeats 10 \
  --conf 0.001 --iou 0.7 --max-det 5
```

engineの精度はexport時に決まるため、`--half` は指定しません。FP16も同じ入力サイズ・元checkpoint・画像リスト・閾値・計測条件で実行します。各結果の `weights_sha256` とengine側の `export.source_weights_sha256` を照合し、異なる学習重みを比較していないことを確認します。

既定のINT8 presetは512×768・raw、既定の長方形FP16 presetはNMS付きです。この2つをそのまま比較すると後処理の場所も違います。数値精度だけを比較するには、同じ寸法でrawのFP16を明示して作れます。

```python
from dataclasses import replace
from axell_export.config import get_preset
from axell_export.exporting import export_engine

config = replace(
    get_preset("fp16-512x768-nms"), preset="fp16-512x768-raw", nms=False,
)
export_engine(
    "/path/private/best.pt", "/path/private/runs/fp16-512x768-raw", config,
)
```

この例では既定presetを上書きし、512×768・rawのFP16を指定しています。実際の設定は `export.json` に保存されます。

NMS付きengineではconfidence・IoUを後から置き換えられません。計測コマンドはexportと同じ閾値、export時以下の検出数上限だけを受け付けます。export記録にNMS閾値がない場合は、固定Ultralytics版の既定 `conf=0.25, iou=0.7, max_det=300` と解釈します。低confidenceまで評価したい場合はraw engineまたは低閾値で再exportしたengineを使います。

```bash
axell-export export \
  --weights /path/private/best.pt --preset fp16-512x768-nms \
  --output-dir /path/private/runs/fp16-low-conf \
  --conf 0.001 --iou 0.7 --max-det 5 --device 0
```

## 計測範囲と出力

最初の画像で指定回数warmupを行った後、全画像を指定回数ずつ推論します。CUDAでは計測開始直前とpredict完了後に `torch.cuda.synchronize(0)` を呼び、非同期処理の完了を待ちます。

| 区分 | 対象 |
| --- | --- |
| 時間に含む | 1画像のpredict呼出し、LetterBox・正規化、GPU転送、モデル推論、必要なNMS、元画像座標への復元、終了同期 |
| 時間に含めない | モデルのロード、warmup、ファイル読込と画像デコード、保存用の出力テンソルのCPU転送、保存用の最終ソート・上位K抽出、JSON書込 |

predict内部のNMS・`max_det`処理は時間に含みます。保存時に行う追加の上位K抽出は時間外です。画像を読む時間を含まないため、これをそのまま公式実行環境の1画像あたり時間やサービス全体のレイテンシとは扱いません。GPU上のカーネルだけの時間とも異なります。

新しい出力ディレクトリに `benchmark.json` を保存します。同じ出力先への上書きは拒否し、途中で失敗した場合も状態と理由を記録します。

- モデル・画像・export記録のSHA-256、モデルのバイト数、設定、Python・ライブラリ・CUDA・デバイス。
- warmupを除いた全サンプル時間、平均・中央値・p95・最小・最大。
- 最初の計測反復の予測：画像ID、元の幅・高さ、クラスID、confidence、元画像のピクセル座標 `xyxy`。保存する予測はconfidence順の上位 `max_det` 件。

モデルごとの出力は同じ画像ID・クラス定義で対応を取って確認します。APの算出、正解ラベルとの照合、公式提出ファイルの生成はこのコマンドには含みません。

## 比較実験の設計

| 比較 | 共通条件 | 変える条件 | 記録したい結果 |
| --- | --- | --- | --- |
| PyTorch / TensorRT FP16 / INT8 | 同じcheckpoint、独立した検証画像、寸法・batch・NMS条件、GPU | バックエンドと精度 | AP、予測の差、時間・容量。APには別途正解データと評価器が必要 |
| calibration | checkpoint、画像サイズ、学習側の候補集合、評価集合 | seed / 抽出方式 / 枚数 | 採用一覧とハッシュ、量子化後のAP・誤り |
| Copy-Paste | train/val、初期重み、seed、epochs、他の拡張 | Copy-Pasteの有無 | 全体AP、対象外物体への誤検出、対象物の見逃し |

APは、同じ画像・ラベルの仕様を持つ評価器で算出します。速度を比較するときはGPU、反復数、入出力サイズと計測範囲をそろえ、測定条件と合わせて記録します。使用できる設定と自動テストの内容は[モデル・変換設定とテスト](experiments.md)に記載しています。

実装の根拠：[固定版のpredictor](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/engine/predictor.py)、[検出の後処理](https://github.com/ultralytics/ultralytics/blob/v8.3.170/ultralytics/models/yolo/detect/predict.py)、[CUDA同期](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)。
