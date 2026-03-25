# 声景频谱事件分析器

一个面向声景音乐、频谱音乐与当代音乐分析的本地原型工具。它的目标不是替代音乐学判断，而是先帮你把“哪里值得重点听、重点看、重点写”自动筛出来。

项目目前提供：

- 音频导入与频谱分析
- 自动事件点检测
- 候选标签生成与人工修订
- 交互式总览、局部试听与局部频谱查看
- 面向论文写作的时间点、事件表与结构化导出

## 适用对象

- 声景音乐研究
- 频谱音乐与当代音乐作品分析
- 需要结合听觉、频谱和时间轴做结构梳理的论文写作
- 想先用机器做一轮事件初筛，再回到人工判断的研究流程

## 当前能力

- 支持 `WAV`、`FLAC`、`AIFF`、`MP3`、`M4A`
- 自动提取 `RMS / spectral centroid / rolloff / flatness / spectral flux / onset strength / novelty`
- 自动找出疑似“新材料进入 / 音色突变 / 结构变化”时间点
- 支持“传统结构模式 / 音色·声景模式”等事件识别预设
- 可点击自动事件点，查看局部试听、局部波形与局部频谱
- 支持人工改标签、写备注、筛选导出
- 支持图形界面与命令行

## 安装

```bash
python3 -m pip install -r requirements.txt
```

## 启动图形界面

```bash
streamlit run app.py
```

启动后，在浏览器里打开终端显示的本地地址，上传音频即可开始分析。

## 命令行使用

```bash
python3 -m spectral_tool.cli your_audio.wav
```

也可以带参数：

```bash
python3 -m spectral_tool.cli your_audio.wav \
  --channel mix \
  --target-sr 22050 \
  --n-fft 4096 \
  --hop-length 1024 \
  --min-event-distance 5 \
  --threshold-sigma 1.0
```

## 输出内容

运行后可得到：

- 事件表 `events.csv`
- 段落建议 `sections.csv`
- 逐帧特征表 `feature_table.csv`
- 结构化结果 `analysis.json`
- 文字摘要 `summary.txt`
- 波形图、频谱图、新颖度曲线、事件密度图

## 项目结构

```text
app.py                  Streamlit 界面入口
requirements.txt        依赖列表
spectral_tool/
  analysis.py           核心分析与事件检测
  assistant.py          AI 小助手与交互层
  cli.py                命令行入口
  visualization.py      绘图与交互可视化
tests/
  test_analysis.py      核心测试
```

## 分析思路

这个工具更接近“自动发现谱形变化”，而不是“自动理解音乐意义”。

核心流程大致是：

1. 把音频转成短时频谱表示
2. 提取能量、频谱重心、滚降、平坦度、谱流量、起音强度等特征
3. 组合成新颖度曲线
4. 在新颖度曲线上做峰值检测
5. 比较事件前后窗口的状态
6. 输出候选标签并允许人工修订

所以它适合回答的是：

- 哪里变了
- 哪里值得重点听
- 哪些点可能是新材料或结构变化

它不应该未经校对直接替代最终论文判断。

## 适合的使用方式

如果你在做《六季》第一乐章《新冰》这类作品分析，可以这样用：

1. 先做一次全曲粗筛，拿到事件时间轴
2. 再回到关键时间点做局部试听和局部频谱核对
3. 把自动事件和人工记录对照
4. 最后整理成论文里的“时间 - 材料 - 结构功能”表

## 局限

- 机器能发现声学变化，但不能自动替代音乐学解释
- 自动标签属于辅助描述，不等于最终结论
- 长时段、极弱动态或高度细腻的纹理仍然需要人工复核

## 测试

```bash
python3 -m unittest discover -s tests
```

## License

本项目使用 [MIT License](LICENSE)。
