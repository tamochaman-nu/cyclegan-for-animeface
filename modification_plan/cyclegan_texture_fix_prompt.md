# CycleGAN テクスチャ学習改善タスク

## 背景・問題

CycleGANで実写→アニメ調変換を学習中。現在、生成結果にアニメの顔パーツ（目・口など）が画像全体に散在する現象が発生している。原因はモデルがテクスチャ（線・塗り・エッジ）ではなく、セマンティックな顔パーツ構造を学習していること。以下の修正を順番に実装すること。

---

## Task 1: 損失の重み調整

**対象ファイル**: `options/base_options.py` または学習スクリプト内のlambda定義箇所

以下の値に変更すること：

```python
lambda_cycle = 5.0      # デフォルト10.0 → 5.0に下げる
lambda_identity = 2.0   # デフォルト0.5 → 2.0に上げる
lambda_adversarial = 2.0
```

**理由**: サイクル損失が支配的になるとGeneratorが顔パーツをステガノグラフィ的に埋め込む。Identity損失を強化することで不要な構造変化を抑制する。

---

## Task 2: Discriminatorの受容野を縮小

**対象ファイル**: `models/networks.py` の `NLayerDiscriminator`

`n_layers` のデフォルト値を変更すること：

```python
# 変更前
def __init__(self, input_nc, ndf=64, n_layers=3):

# 変更後
def __init__(self, input_nc, ndf=64, n_layers=1):
```

受容野の対応：
- `n_layers=3` → 受容野70x70（構造・パーツを見る）
- `n_layers=1` → 受容野16x16（局所テクスチャのみを見る）

---

## Task 3: Perceptual Loss（浅い層）を追加

**対象ファイル**: `models/cycle_gan_model.py`

以下のクラスを追加し、Generator損失に組み込むこと：

```python
import torchvision.models as models
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        # 浅い層のみ使用（深い層はセマンティック情報を含むため使わない）
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # conv1_2（エッジ）
        self.slice2 = nn.Sequential(*list(vgg.children())[:9])   # conv2_2（色・塗り）
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        loss += F.l1_loss(self.slice1(x), self.slice1(y)) * 1.0
        loss += F.l1_loss(self.slice2(x), self.slice2(y)) * 0.5
        return loss
```

`cycle_gan_model.py` の `__init__` でインスタンス化：

```python
self.perceptual_loss = VGGPerceptualLoss().to(self.device)
self.lambda_perceptual = 1.0
```

`backward_G` 内に追加：

```python
self.loss_perceptual = self.perceptual_loss(self.fake_B, self.real_A) * self.lambda_perceptual
self.loss_G += self.loss_perceptual
```

---

## Task 4: Gram Matrix Loss を追加

**対象ファイル**: `models/cycle_gan_model.py`（Task 3と同じファイル）

Gram行列は空間位置情報を破棄するため、「パーツがどこにあるか」が損失に影響しなくなる。

以下の関数を追加すること：

```python
def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

def gram_style_loss(x, y, vgg_slice):
    fx = vgg_slice(x)
    fy = vgg_slice(y)
    return F.l1_loss(gram_matrix(fx), gram_matrix(fy))
```

`backward_G` 内に追加：

```python
self.loss_gram = gram_style_loss(self.fake_B, self.real_A, self.perceptual_loss.slice1) * 0.5
self.loss_G += self.loss_gram
```

---

## Task 5: データローダーにパッチクロップを追加

**対象ファイル**: `data/unaligned_dataset.py`

transformsにランダムクロップを追加すること。顔全体が1枚の画像に収まりにくくなり、テクスチャ学習に集中しやすくなる。

```python
from torchvision import transforms

transform_list = [
    transforms.RandomCrop(128),          # 追加: 顔全体が映りにくくなる
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),     # 追加: 顔の向きの概念を破壊
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
```

---

## Task 6: 損失ログの追加（診断用）

**対象ファイル**: `models/cycle_gan_model.py`

学習ループ内に以下のログ出力を追加し、損失バランスを監視できるようにすること：

```python
print(f"[Epoch {epoch}] "
      f"loss_cycle: {self.loss_cycle_A:.4f} | "
      f"loss_G_GAN: {self.loss_G_A:.4f} | "
      f"loss_identity: {self.loss_idt_A:.4f} | "
      f"loss_perceptual: {self.loss_perceptual:.4f}")
```

**確認ポイント**: `loss_cycle >> loss_G_GAN` の状態が続く場合、Task 1のlambda調整が不十分。`loss_cycle : loss_G_GAN` がおおよそ `2:1 〜 3:1` になるのが目安。

---

## 実装優先順位

| 優先度 | タスク | 変更規模 |
|--------|--------|----------|
| 最高   | Task 1: lambda調整 | 数値変更のみ |
| 高     | Task 2: Discriminator受容野縮小 | 1行変更 |
| 中     | Task 3: Perceptual Loss追加 | クラス追加 |
| 中     | Task 4: Gram Matrix Loss追加 | 関数追加 |
| 低     | Task 5: パッチクロップ | transform変更 |
| 低     | Task 6: ログ追加 | print追加 |

Task 1 → Task 2 の順に適用し、各ステップで数エポック学習して効果を確認してから次のタスクに進むこと。
