# 使用文档

本文档提供 PlanetaryFeatureMatch 的常用命令示例。更详细的训练说明见 `docs/training.md`。

## 构建

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

## 查看帮助

```bash
./build/pfm_cli --help
./build/pfm_cli train --help
./build/pfm_cli extract --help
./build/pfm_cli match --help
./build/pfm_cli eval --help
./build/pfm_cli export --help
```

## 训练

```bash
./build/pfm_cli train \
  --image-dir images \
  --checkpoint model.pt \
  --epochs 1 \
  --batch-size 1
```

## 特征提取

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

## 图像匹配

```bash
./build/pfm_cli match \
  --image-a images/a.tif \
  --image-b images/b.tif \
  --checkpoint model.pt \
  --output matches.pt \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

## 批量评估

`pairs.txt` 每行包含一对图像路径：

```text
images/a.tif images/b.tif
"/path/with spaces/a.tif" "/path/with spaces/b.tif"
```

运行：

```bash
./build/pfm_cli eval \
  --pairs pairs.txt \
  --checkpoint model.pt \
  --output report.pt \
  --max-keypoints 1024
```

## 导出

```bash
./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

## 输出文件

- `model.pt`：训练 checkpoint。
- `features.pt`：单图特征。
- `matches.pt`：双图匹配结果。
- `report.pt`：评估报告。
- `exported.pt`：导出后的推理 checkpoint。
