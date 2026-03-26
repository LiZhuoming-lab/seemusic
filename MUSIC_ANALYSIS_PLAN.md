# 音乐分析平台升级方案

## 目标

把现有“声景 / 频谱事件分析器”升级为一个同时支持：

- 音频分析
- 乐谱 / 符号分析
- 后续音频 + 乐谱联合分析

的音乐分析平台。

当前策略是不推翻现有音频工作流，而是在现有项目上增量加入新的分析层。

## 当前代码架构映射

- [app.py](/Users/dimash/Documents/New project/app.py)
  统一的 Streamlit 工作台入口。当前已支持“音频 / 频谱分析”和“乐谱 / 符号分析”双工作空间。
- [spectral_tool/analysis.py](/Users/dimash/Documents/New project/spectral_tool/analysis.py)
  现有音频特征提取、事件检测、候选边界分析核心。
- [spectral_tool/assistant.py](/Users/dimash/Documents/New project/spectral_tool/assistant.py)
  音频事件的交互式解释与辅助判断层。
- [spectral_tool/visualization.py](/Users/dimash/Documents/New project/spectral_tool/visualization.py)
  音频可视化层。
- [spectral_tool/symbolic_analysis.py](/Users/dimash/Documents/New project/spectral_tool/symbolic_analysis.py)
  新增的乐谱 / 符号分析核心层。

## 分层方案

### Layer A: Score / Symbolic

职责：

- MusicXML / MXL / MIDI 导入
- note / chord / measure / part 级索引
- 音高统计
- 音程序类统计
- 和声切片与 Roman numeral 候选
- 主题 / 动机再现候选

当前实现位置：

- [spectral_tool/symbolic_analysis.py](/Users/dimash/Documents/New project/spectral_tool/symbolic_analysis.py)

### Layer B: Audio

职责：

- RMS、centroid、rolloff、flatness、flux、onset 等音频特征
- novelty 检测
- 候选事件与结构边界
- 局部试听与交互式修订

当前实现位置：

- [spectral_tool/analysis.py](/Users/dimash/Documents/New project/spectral_tool/analysis.py)
- [spectral_tool/assistant.py](/Users/dimash/Documents/New project/spectral_tool/assistant.py)
- [spectral_tool/visualization.py](/Users/dimash/Documents/New project/spectral_tool/visualization.py)

### Layer C: Integrated

目标：

- score event 与 audio event 对齐
- 将乐谱结构与声音实现进行比较
- 支持“同一作品的谱面分析 + 声学实现分析”

当前状态：

- 还未实现
- 适合在 Phase 2 之后进入

## 新增数据结构

### `SymbolicAnalysisConfig`

位置：

- [spectral_tool/symbolic_analysis.py](/Users/dimash/Documents/New project/spectral_tool/symbolic_analysis.py)

字段：

- `theme_window_notes`
- `max_recurrence_results`
- `measure_summary_top_n`

作用：

- 控制主题检索窗口
- 控制候选输出数量
- 控制小节摘要密度

### `note_table`

粒度：单个音高事件

字段示例：

- `part_name`
- `measure_number`
- `beat`
- `offset_ql`
- `quarter_length`
- `pitch_name`
- `midi`
- `pitch_class`
- `scale_degree`

### `harmony_table`

粒度：和声切片

字段示例：

- `slice_id`
- `measure_number`
- `beat`
- `pitch_names`
- `pitch_class_set`
- `root`
- `root_scale_degree`
- `quality`
- `roman_numeral`

### `interval_table`

粒度：连续旋律音之间的关系

字段示例：

- `part_name`
- `from_measure`
- `to_measure`
- `from_pitch`
- `to_pitch`
- `semitones`
- `interval_class`
- `directed_name`
- `contour`

### `theme_matches`

粒度：主题 / 动机再现候选

字段示例：

- `relation_type`
- `similarity_score`
- `source_part`
- `source_measure`
- `source_excerpt`
- `match_part`
- `match_measure`
- `match_excerpt`
- `transposition_semitones`
- `match_detail`

## 分阶段开发顺序

### Phase 1: 最小可用的符号分析层

已完成：

- MusicXML / MXL / MIDI 导入
- 音高统计
- 音级类统计
- 音程 / 音程序类统计
- 和声切片
- Roman numeral 候选
- 连续音窗口的主题再现检索
- Streamlit 中的人工修订入口

### Phase 2: 更强的分析工作流

建议下一步做：

1. 局部调性 / 中心音估计
2. 分段 pitch inventory 对比
3. 主旋律选择与多声部筛选
4. 和声节奏摘要
5. 用户可编辑段落 / 乐句边界

### Phase 3: 动机与形式

建议做：

1. contour-based approximate match 提升
2. rhythm compression / expansion 检测
3. fragmentation / embedded motive 检测
4. 乐句与段落级摘要
5. section comparison 视图

### Phase 4: 联合分析

建议做：

1. MIDI / MusicXML 与音频时间轴对齐
2. score event 对 audio event 映射
3. 乐谱结构与声学结构比较
4. 写作导向的联合报告导出

## 当前第一里程碑

### 已交付

1. 在现有项目中保留原有音频分析能力
2. 新增乐谱 / 符号分析工作空间
3. 新增核心模块：
   - [spectral_tool/symbolic_analysis.py](/Users/dimash/Documents/New project/spectral_tool/symbolic_analysis.py)
4. 新增测试：
   - [tests/test_symbolic_analysis.py](/Users/dimash/Documents/New project/tests/test_symbolic_analysis.py)

### 当前可回答的问题

- 这首作品总体用了多少音高与多少音级类？
- 哪些 pitch class 最突出？
- 哪些音程 / 音程序类最常见？
- 哪些位置形成了明显的和声切片？
- 哪些和声切片可能对应 `I / IV / V` 等级数候选？
- 某个旋律窗是否在后面精确再现或移调再现？

### 当前边界

- Roman numeral 仍是候选，不是最终定论
- 主题再现匹配目前是第一阶段窗口法，不是最终音乐学判断
- 尚未进入局部调性、乐句、形式和音频-乐谱联合分析
