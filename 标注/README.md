# 匹配标注器

这个目录放图像对数据和人工匹配点标注。原生桌面 GUI 启动：

```bash
python3 /home/xjw/code/deeplearning/PlanetaryFeatureMatch/标注/annotator_gui.py
```

数据组织方式任选一种：

```text
标注/
  case_001/
    left.tif
    right.tif
  case_002/
    a.tif
    b.tif
```

或者写 `pairs.json`：

```json
{
  "pairs": [
    {
      "id": "case_001",
      "image_a": "case_001/left.tif",
      "image_b": "case_001/right.tif"
    }
  ]
}
```

标注结果自动保存到 `标注/annotations/<pair_id>.json`，坐标是原始图像像素坐标。

操作方式：

- 点击左右面板上的“选择影像”可以手动指定左图和右图；影像需位于当前 `--root` 标注根目录内。
- 鼠标左键拖动影像；鼠标中键用于添加、确认、选中和拖动标注点。
- 中键点左图一个位置后，工具会根据已经保存/当前页面中的匹配点预测右图位置。
- 每次在影像上成功中键标点都会闪出黄色高亮圈，并在状态栏显示点击坐标和当前动作。
- 右图预测点是橙色十字；中键点击它会确认，中键拖动它后松开会用微调后的位置确认。
- 也可以在右图直接中键点一个位置，覆盖预测并确认。
- 中键点击已有点或点击匹配点列表行可以选中点对，然后点击“删除点对”删除。
- `Enter` 确认当前预测，`Esc` 取消，`Delete` 删除选中的匹配点。
