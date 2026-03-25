# 声景频谱事件分析器

这是一个面向声景音乐与频谱分析的原型软件，用来自动标记音频中疑似“新频谱进入”“音色结构突变”或“材料层次切换”的时间位置。它适合拿来辅助类似《六季》第一乐章《新冰》这样的作品分析，把原本完全依赖人工听辨的工作，先交给机器做一轮初筛。

## 这套工具能做什么

- 读取音频并计算短时频谱图
- 自动提取 `RMS / centroid / rolloff / flatness / flux / onset strength / novelty`
- 自动找出疑似新材料出现的时间点
- 自动生成候选标签，例如“高频扩展、噪声侵入、材料聚集、消散、段落转换”
- 点击自动标记点，显示精确时间
- 支持局部试听与人工修订标签
- 输出事件表、粗分段建议、特征表和文字摘要
- 支持图形界面与命令行批量导出

## 适用场景

- 声景音乐结构分析
- 频谱音乐、电子音乐或混合媒介作品的初步分段
- 辅助标记“哪里值得重点听”
- 为论文写作建立时间轴、事件表和图示材料

## 运行方式

先安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

启动图形界面：

```bash
streamlit run app.py
```

启动后，在浏览器中上传音频即可分析。支持 `WAV`、`FLAC`、`AIFF`、`MP3` 和 `M4A`，其中 `WAV` 仍然最稳定。

## 交互功能

- 在“交互分析”页点击自动标记点，可以直接查看该事件的精确时间
- 点击后右侧会同步显示候选标签、规则摘要和局部试听
- 你可以手动修改事件标签、添加备注，并决定是否导出该事件
- 段落边界也支持单独人工修订

## 命令行批量分析

```bash
python3 -m spectral_tool.cli your_audio.wav
```

也可以自定义参数：

```bash
python3 -m spectral_tool.cli your_audio.wav \
  --channel mix \
  --target-sr 22050 \
  --n-fft 4096 \
  --hop-length 1024 \
  --min-event-distance 5 \
  --threshold-sigma 1.0
```

结果会导出到 `exports/音频文件名/`：

- `events.csv`：自动事件表
- `sections.csv`：粗分段建议
- `feature_table.csv`：逐帧特征表
- `analysis.json`：结构化结果
- `summary.txt`：文字摘要
- `waveform_events.png`：波形与事件图
- `novelty_curve.png`：新颖度曲线图
- `spectrogram.png`：频谱图
- `event_density.png`：事件密度图

## 算法思路

这不是一个“自动理解音乐意义”的系统，而是一个“自动发现谱形变化”的系统。核心步骤如下：

1. 把音频转成短时频谱表示
2. 自动提取 `RMS / centroid / rolloff / flatness / flux / onset strength`
3. 计算相邻频谱帧之间的差异，并结合能量变化生成新颖度曲线
4. 在新颖度曲线上做峰值检测，得到疑似新事件
5. 比较事件前后窗口的频谱特征，生成描述语
6. 依据规则给出候选标签，并允许人工修订

输出中的文字标签，例如“低频能量增强”“频谱重心上移”“噪声质地增强”，属于机器辅助描述，目的是帮助你快速定位和整理，不建议未经校对直接当成最终论文结论。

## 参数建议

- 事件太少：降低 `threshold_sigma`，减小 `min_event_distance`
- 事件太密：提高 `threshold_sigma`，增大 `min_event_distance`
- 想看更细微变化：减小 `hop_length`
- 想看更稳定的宏观结构：增大 `min_event_distance` 和 `context_window`
- 想更清晰地区分事件点击位置：优先看“交互分析”页中的可点击新颖度图

## 对你的研究尤其有用的用法

如果你后面要继续分析《新冰》这类作品，可以这样做：

1. 先用混合通道做一次粗筛，得到全曲事件时间轴
2. 再分别分析左声道和右声道，观察声场分工是否不同
3. 把 `events.csv` 和你已有的人工听觉记录对照
4. 最后把自动事件和人工判断结合，形成论文中的“时间-材料-结构功能”表

## 局限

- 它能自动发现“哪里变了”，但不能完全替代音乐学判断
- 它更擅长发现声学突变，不一定能准确理解作品中的文化语义
- 对非常长、非常安静或层次极细的音频，参数需要手动调整

## 测试

```bash
python3 -m unittest discover -s tests
```
