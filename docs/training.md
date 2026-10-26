# 対象外物体のCopy-Pasteを使った学習

この手順では、利用する権利のある自分の画像と素材を用意し、対象外の缶などを追加する拡張をYOLOの学習に組み込みます。追加物体へ正解ラベルを付けず、元の正解枠を避けて配置します。考え方は[解法と設計](solution.md)を参照してください。

対応環境はUltralytics 8.3.170の物体検出、単一プロセス・単一デバイスです。複数のデータローダーworkerを使用できますが、複数GPUを使う分散学習は対象外です。

## 環境を用意する

CPUで合成画像のテストを実行する場合は、[README](../README.md)のPython 3.12用CPU lockを使用します。GPUで学習する場合は、CUDA対応PyTorchとruntime依存関係を導入します。TensorRTは学習には不要です。

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[runtime]'
```

以下の例はランダム初期化したモデルを学習する構成です。事前学習checkpointや配布データの取得処理は含みません。各パスを自分の非公開ディレクトリへ置き換えてください。

## 学習画像と検証画像を分ける

以下は、自分で用意した2クラスのデータの例です。競技の配布画像をコンペ外の動作確認や学習に使用しないでください。素材も含め、他コンペのデータや成果物を流用しません。

```text
/path/private/owned-dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   └── sample.png
│   └── val/
│       └── independent.png
└── labels/
    ├── train/
    │   └── sample.txt
    └── val/
        └── independent.txt
```

`data.yaml`は次の形式です。検証には学習画像と重複しない独立した画像を置きます。分割の生成は利用者が行います。

```yaml
path: /path/private/owned-dataset
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

ラベルはYOLOの物体検出形式で、1行を`class_id center_x center_y width height`とします。座標・幅・高さは画像サイズで正規化した値です。クラス番号はこの`names`に対応します。

```text
0 0.5 0.5 0.2 0.4
```

この学習用YAMLには独立した`val`が必要です。INT8 calibrationで使う、`train`と`val`が同じ画像を指す専用YAMLを学習へ流用しないでください。

## 貼り付ける素材を明示する

素材の一覧を、たとえば`/path/private/negative-assets.yaml`へ保存します。画像とラベルを1組ずつ指定し、`negative_class_ids`に素材側で採用するクラスを列挙します。

```yaml
negative_class_ids: [0, 2]
assets:
  - image: owned-assets/train/can-a.png
    label: owned-assets/train/can-a.txt
  - image: owned-assets/train/can-b.png
    label: owned-assets/train/can-b.txt
```

相対パスはこの一覧ファイルの場所を基準に解決します。素材のラベルもYOLO形式です。指定したクラスの矩形だけを切り出し、その他のクラスは素材に採用しません。ディレクトリを自動走査したり、画像をダウンロードしたりする処理はありません。

ここでの`0`・`2`は**素材側の分類**であり、学習対象の`class_a`・`class_b`や競技の91クラスの番号とは対応しません。選んだ矩形に学習対象の物体・銘柄が写っていないことを確認してください。対象物体をラベルなしで貼り付けると、学習に矛盾が生じます。検証・テスト画像から素材を抽出せず、学習に利用できる権利のある画像から選びます。

一覧の未知キー、重複、URL、不正なラベルはエラーにします。素材の読み込みでは元画像・ラベルを上書きせず、`crop_inset=0.1`なら矩形の各辺から5%を除いた範囲を切り出します。

## モデル名と拡張設定を指定して学習する

次のコードを自分の実行用スクリプトとして保存します。`yolo11n`を[登録済みのモデル名](../README.md)へ変更すると構成を選べます。出力クラス数は学習用YAMLから設定されます。

```python
from functools import partial

from axell_export.assets import load_negative_assets
from axell_export.augmentation import CopyPasteConfig
from axell_export.registry import get_model_spec
from axell_export.training import CopyPasteTrainer
from ultralytics import YOLO


def main():
    config = CopyPasteConfig(
        probability=0.5,
        max_pastes=4,
        max_attempts=50,
        crop_inset=0.1,
        area_mode="mean",
    )
    negative_assets = load_negative_assets(
        "/path/private/negative-assets.yaml", config
    )
    trainer = partial(
        CopyPasteTrainer,
        negative_assets=negative_assets,
        copy_paste_config=config,
        copy_paste_stop_last_epochs=10,
    )
    model = YOLO(get_model_spec("yolo11n").yaml_name)
    model.train(
        trainer=trainer,
        data="/path/private/owned-dataset/data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device=0,
        workers=8,
        project="/path/private/runs",
        name="yolo11n-negative-copy-paste",
        exist_ok=False,
        pretrained=False,
        plots=False,
        amp=False,
        seed=42,
    )


if __name__ == "__main__":
    main()
```

`probability`は画像ごとに追加処理を試みる確率です。追加個数は1〜`max_pastes`から選び、正解枠や既に追加した素材と重ならない配置を探します。配置できなければ見送るため、指定個数を必ず追加する設定ではありません。`area_mode="jitter"`にすると、正解枠の面積を基準に大きさを変化させます。詳細は[サイズと適用頻度の設定](solution.md#サイズと適用頻度の設定)にあります。

この例では50 epoch中、最後の10 epochでCopy-Pasteを無効化します。確率変更時にworkerを作り直し、変更前に先読みしたバッチを破棄します。`copy_paste_stop_last_epochs=0`なら終盤も設定した確率を維持します。検証画像にはCopy-Pasteを適用しません。

学習の乱数seedはUltralyticsが各workerへ設定します。worker数・実行環境を変えたときに同一画像が同一の順序で生成されることまでは保証しません。比較する実験では、データ分割、seed、worker数、モデル・学習設定もそろえて記録します。

## 保存物と実行時の動作

checkpointや学習記録は指定した`project`以下へ保存されます。モデル・画像・ラベル・生成物は非公開ディレクトリで管理してください。素材loaderとCopy-Pasteは入力を変更しませんが、標準の学習データセット処理はラベルcacheを作り、壊れたJPEGを補修することがあります。学習には元データを保管したうえで作業コピーを指定します。

また、Ultralyticsのデータ検査はフォントを取得する場合があります。`plots=False`でも完全なオフライン動作を保証する手順ではありません。CPUテストではフォント確認を置き換え、画像や事前学習重みをダウンロードせずに学習処理を検証しています。

MosaicなどUltralytics標準の拡張は、その設定に従って動きます。Copy-Pasteはそれらの後、テンソル化の直前に実行し、リサイズした矩形の画素を貼り付けます。

素材の選択、貼り付け、workerへの反映、終盤の無効化を確認する自動テストは[モデル・変換設定とテスト](experiments.md)にまとめています。
