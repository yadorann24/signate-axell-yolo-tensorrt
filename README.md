# Axell YOLO → TensorRT

**Axell AI Contest 2025「飲料パッケージの物体検出」最終35位・銅メダル。** yadorann24として個人参加しました。検出対象外の缶を負例として加える工夫と、精度・推論時間・容量の制約を踏まえたYOLOのTensorRT変換を扱っています。

| 競技提出の公式結果 | 値 |
| --- | ---: |
| 最終順位・メダル | 35位・bronze |
| AP@[.50:.95] | **0.9674124** |
| 推論時間 | **16.470296 ms / 画像** |

[公式最終リーダーボード](https://user.competition.signate.jp/ja/competition/detail/?competition=6e644a0277314ba2a1d36a1634f75f41&task=cc969d0e55a44990becf0c0688235765&tab=leaderboard&leaderboard=private)の値です。

8種類のモデル構成と9種類の変換設定を名前で選択し、学習・変換・推論の条件と成果物を追跡できる実装です。

[解法と設計](docs/solution.md)では、**検出対象外の缶などを画像に追加するCopy-Paste**の狙い、正解物体を隠さない配置、サイズの調整、ラベルの扱いを説明しています。[学習の手順](docs/training.md)から、利用する権利のある自分の画像で実装を試せます。

実行できるコードは、モデル構成の生成、Copy-Pasteを組み込んだ学習、変換設定の確認、ローカルcheckpointからのexport、利用権限のある画像での推論・時間計測です。公式提出用の処理やAP評価器は含みません。データセット、貼り付け素材、学習済み重み、生成したengineは同梱していません。

## 課題と設計上の関心

課題は、画像内の飲料の位置と91種類の銘柄を検出することです。公式評価はAP@[.50:.95]で、各画像の信頼度上位5件を使用します。投稿ファイル全体に64 MB、推論に17.5 ms / 画像の制約があり、モデルの大きさ、入力解像度、数値精度、後処理を合わせて検討する必要があります。[公式競技ページ](https://user.competition.signate.jp/ja/competition/detail/?competition=6e644a0277314ba2a1d36a1634f75f41&task=cc969d0e55a44990becf0c0688235765)

この実装では、検討した条件を比較しやすい形で保存し、変換の取り違えを防ぐことを重視しています。

- **設定を名前で選ぶ**：モデル構成とexport設定を別々に管理し、実行する引数を事前に確認できます。
- **負例を学習バッチへ反映する**：対象外クラスの素材だけを明示的に選び、正解枠を避けて貼り付けます。worker起動、Mosaic終了、学習終盤の無効化に対応します。
- **既存の成果物を保護する**：新しい出力ディレクトリで処理し、同名engineや古いcalibration cacheの流用を避けます。
- **INT8の入力を追跡する**：指定した学習画像から固定seedで抽出し、選択した画像の内容ハッシュを保存します。512 × 768の長方形入力を保ち、原画像やラベルを書き換えません。
- **測定条件をそろえる**：NMSのconfidence・IoU・検出数上限を保存し、推論計測では入力サイズ・warmup・反復数とGPU同期を明示します。
- **実行環境を分ける**：設定の確認やモデル構成の生成はCPUで実行でき、TensorRTへの変換にはGPU用の環境を使います。


## 最初に試す

Python 3.11または3.12を使用します。設定一覧と実行計画の確認にはCUDAやPyTorchは不要です。

```bash
git clone https://github.com/yadorann24/signate-axell-yolo-tensorrt.git
cd signate-axell-yolo-tensorrt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

axell-export list-models
axell-export list-presets
axell-export describe-preset --preset fp16-640-nms
```

手元のcheckpointを指定して、変換前に設定を確認できます。以下のパスは自分の環境に置き換えてください。出力先には、まだ存在しない非公開ディレクトリを指定します。

```bash
axell-export plan \
  --weights /path/private/best.pt \
  --preset fp16-640-nms \
  --output-dir /path/private/runs/fp16-640 \
  --device 0
```

`plan`はTensorRT engineを生成しません。依存ライブラリやGPUが実際の変換に対応しているかは、export実行時にも確認します。

## モデル名から構成を作る

モデル生成を試す場合は、PyTorchを含むruntime依存関係を導入します。以下のCPU lockは**Linux x86_64・CPython 3.12用**です。この環境ではPython 3.12で仮想環境を作成してから実行してください。他の環境では`python -m pip install -e '.[runtime]'`で導入します。

```bash
python -m pip install --require-hashes -r requirements-cpu.lock
python -m pip install --no-deps -e '.[runtime]'
```

```python
from axell_export import build_model

model = build_model("yolo11m", num_classes=91)
```

これはランダムに初期化された物体検出モデルの構成を作ります。事前学習重みはダウンロードせず、このモデルだけでは飲料を検出できません。TensorRTへの変換には、自分で用意した学習済みcheckpointを使用します。

| 系列 | 使用できるモデル名 |
| --- | --- |
| YOLO11 | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` |
| YOLO12 | `yolo12n`, `yolo12m`, `yolo12l` |

## 対象外の缶を追加して学習する

`CopyPasteConfig`で適用確率・追加個数・サイズを設定し、`CopyPasteTrainer`を`YOLO.train()`へ渡します。学習画像だけに対象外物体を追加し、追加物体へ正解ラベルは付けません。検証画像にはこの処理を適用しません。

素材はローカルのYAMLで画像・ラベル・採用するクラス番号を列挙します。元画像を上書きせず、既存の正解枠と重ならない位置へ矩形を貼り付けます。素材のクラス番号は素材側の分類であり、検出対象のクラス番号とは別です。素材が検出対象を含まないことと、学習・検証画像を含めた利用権限を確認して用意してください。

環境構築、データ形式、実行コード、学習終盤で無効化する設定は[学習の手順](docs/training.md)にまとめています。対応範囲は単一プロセス・単一デバイスです。複数workerで学習バッチへ反映されることは合成画像で検証しています。

## GPUでTensorRTへ変換する

NVIDIA GPUと対応するドライバが必要です。変換コマンドは`--device 0`に対応します。CPU用環境とは分けて環境を作り、CUDA対応PyTorchとexport依存関係を導入してください。以下はLinux向けの構成です。PyTorchの配布元は[公式の過去バージョン一覧](https://pytorch.org/get-started/previous-versions/#v240)で確認できます。

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[export]'

axell-export export \
  --weights /path/private/best.pt \
  --preset fp16-640-nms \
  --output-dir /path/private/runs/fp16-640 \
  --device 0
```

生成した`model.engine`と実行記録の`export.json`は、指定した非公開ディレクトリに保存します。実行記録には、要求した設定、実際に渡した引数、依存ライブラリとGPUの情報、checkpointとengineのSHA-256、処理の成功・失敗を残します。同じ出力先への再実行は受け付けないため、条件を変える場合や失敗後の再試行では新しいディレクトリ名を指定してください。

FP16 / INT8は変換時に要求する設定です。実際に使われる演算精度はGPUとTensorRTの対応状況にも依存します。

INT8の例は次のとおりです。calibration用YAMLには、利用する権利のあるローカルの学習画像を指定してください。

```bash
axell-export plan \
  --weights /path/private/best.pt \
  --preset int8-512x768-raw \
  --calibration-data /path/private/calibration.yaml \
  --calibration-selection seeded-random --calibration-seed 42 \
  --output-dir /path/private/runs/int8-512x768 \
  --device 0
```

確認後に`plan`を`export`へ変更すると変換を実行します。calibrationの入力例と処理の説明は[解法と設計](docs/solution.md)を参照してください。

## 推論時間と出力を比較する

`axell-benchmark`は、明示したローカル画像に対してPyTorch checkpointまたはTensorRT engineで推論し、warmup後の平均・中央値・p95、モデル容量、入力画像とモデルのハッシュ、元画像座標の予測枠を保存します。GPUでは計測前後に同期します。

```bash
axell-benchmark \
  --weights /path/private/best.pt \
  --images /path/private/evaluation-images.txt \
  --output-dir /path/private/benchmarks/pytorch \
  --image-size 512 768 --device 0 --warmup 10 --repeats 10 \
  --conf 0.001 --iou 0.7 --max-det 5
```

同じcheckpointからのFP16 / INT8との比較手順、engine内NMSの扱い、計測に含む範囲は[推論・ベンチマーク](docs/benchmark.md)に記載しています。このコマンドは公式評価器ではなく、精度改善や17.5 ms制約の達成を自動判定しません。

## 読み進める

| 文書 | 内容 |
| --- | --- |
| [解法と設計](docs/solution.md) | 対象外の缶を追加するCopy-Paste、モデル・入力解像度・数値精度・NMS、変換フロー |
| [学習の手順](docs/training.md) | 自分の画像での学習、素材一覧とクラス選択、終盤の無効化 |
| [モデル・変換設定とテスト](docs/experiments.md) | 9種類の設定、登録モデル、依存環境と自動テスト |
| [推論・ベンチマーク](docs/benchmark.md) | 同じ画像と条件でのPyTorch / TensorRT比較、計測範囲、追加実験の設計 |
| [使用ライブラリと出典](docs/provenance.md) | 依存元、公式資料、ライセンス |

コードは`src/axell_export/`、テストは`tests/`にあります。配布画像・学習済み重みは使用せず、CPUで8モデルへの合成入力によるforward、Copy-Pasteとworker、calibration、NMS設定、exportの制御、ランダム初期化モデルの推論計測を検証します。テスト項目は[モデル・変換設定とテスト](docs/experiments.md)に記載しています。

```bash
python -m unittest discover -s tests
```

コードのライセンスは[AGPL-3.0](LICENSE)です。依存ライブラリの出典と利用条件は[使用ライブラリと出典](docs/provenance.md)に記載しています。

SIGNATE配布画像は同梱せず、再配布やコンペ外の利用を行いません。コードを試す場合は、自分で用意した画像と、利用権限・出所を確認できるcheckpointを使用してください。データ・学習済み重み・生成物はリポジトリ外で管理します。
