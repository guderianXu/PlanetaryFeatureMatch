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
  --batch-size 1 \
  --device cpu
```

## 特征提取

```bash
./build/pfm_cli extract \
  --image images/a.tif \
  --checkpoint model.pt \
  --output features.pt \
  --device cpu \
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
  --device cpu \
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
  --device cpu \
  --max-keypoints 1024 \
  --semi-dense-threshold 0.5
```

## 导出

```bash
./build/pfm_cli export \
  --checkpoint model.pt \
  --output exported.pt
```

## CUDA 设备

所有训练/推理命令默认使用 `--device cpu`。如果 LibTorch 是 CUDA 版本，可以使用：

```bash
--device cuda
--device cuda:0
```

`cuda` 等价于 `cuda:0`。CUDA 不可用、索引越界或格式错误时会明确失败，不会静默退回 CPU。当前 CUDA 覆盖训练 forward/backward/loss 和推理模型 forward；图像读取、特征解码、匹配后处理、评估汇总和 `.pt` 写出仍在 CPU。

## 测试程序 `pfm_tests`

`./build/pfm_tests` 是项目的 C++ 单元测试运行器。它会逐个运行模块测试并输出 `PASS <test_name>`，所以看到很多 `PASS` 是正常的。最后一行 `N test(s) passed` 且退出码为 0 表示全部通过；如果失败，会输出 `FAIL <test_name>: <reason>` 并返回非 0。

也可以运行：

```bash
ctest --test-dir build --output-on-failure
```

## 输出文件

- `model.pt`：训练 checkpoint。
- `features.pt`：单图特征。
- `matches.pt`：双图匹配结果。
- `report.pt`：评估报告。
- `exported.pt`：导出后的推理 checkpoint。
