"""Auditable Task 2A.3 tables, figure, and 30-question summary."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fmt(value):
    if isinstance(value,(float,np.floating)):return f'{value:.6f}' if np.isfinite(value) else 'NA'
    return str(value).replace('|','/')


def _table(frame: pd.DataFrame) -> str:
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(_fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def write_multiscale_report(paths,config,protocol,statistics,tables):
    coverage=tables['coverage'];random_cov=tables['coverage_random'];random_reg=tables['region_random']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for detector,group in coverage.groupby('detector'):
        axes[0].plot(group.actual_coverage*100,group.overall_recall*100,marker='o',label=detector)
    axes[0].set(xlabel='Actual genomic coverage (%)',ylabel='Overall recall (%)',title='Equal-coverage detector comparison')
    multi=coverage.loc[coverage.detector.eq('multiscale')]
    for label in ('CHIN','OPCID','CHID'):
        axes[1].plot(multi.actual_coverage*100,multi[f'{label}_recall']*100,marker='o',label=label)
    axes[1].set(xlabel='Actual genomic coverage (%)',ylabel='Recall (%)',title='MultiScale per-class recall')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.suptitle(f'Task 2A.3 multi-scale candidate detector ({config.mode})');fig.tight_layout()
    fig.savefig(paths.output_root/'multiscale_coverage_recall.png',dpi=160);plt.close(fig)
    summary=statistics['scale_summary'];readiness=statistics['readiness'];rescue=tables['rescue'];audit=tables['known_audit']
    selected=coverage.loc[coverage.budget.isin([.05,.10,.20,.30])]
    lines=['# Task 2A.3 multi-scale candidate detector summary','',
        '> SMOKE：仅验证工程链路，不支持科学结论。' if config.mode=='smoke' else '> FULL：所有尺度、融合、NMS、预算、随机基线与 readiness 均在 known recall 前冻结。','',
        '## 1. 为什么做 multi-scale？','',
        '单一 6.4 kb 窗口可能稀释较短 CHIN，又无法完整表达较长 CHID；本轮只改变物理窗口尺度，100 bp bin resolution 保持不变。','',
        '## 2. 三个尺度是什么？','',
        '`small=3200/400`、`medium=6400/800`、`large=12800/1600`（window/step bp），矩阵为32/64/128。','',
        '## 3. 各尺度窗口数','',_table(summary[['scale','n_windows']]),'',
        '## 4. 各尺度 Background 数量','',_table(summary[['scale','train_background','validation_background','calibration_background']]),'',
        '## 5. 所有 background known overlap 是否为0？','',
        f"`{statistics['known_overlap_zero']}`。严格使用完整 `[structure_start,structure_end)`，margin=0；详见 background_overlap_audit.csv。",'',
        '## 6. 是否存在跨尺度 split leakage？','',f"`{not statistics['no_leakage']}`；审计结论 no_leakage=`{statistics['no_leakage']}`。",'',
        '## 7. Small AE','',_table(summary.loc[summary.scale.eq('small'),['best_epoch','best_validation_loss']]),'',
        '## 8. Medium AE','',_table(summary.loc[summary.scale.eq('medium'),['best_epoch','best_validation_loss']]),'',
        '## 9. Large AE','',_table(summary.loc[summary.scale.eq('large'),['best_epoch','best_validation_loss']]),'',
        '## 10. 每尺度 Calibration sample size','',_table(summary[['scale','calibration_background','minimum_empirical_p']]),'',
        '经验上尾固定为 `(1+count(cal>=x))/(n+1)`，每个 scale/replicate/branch 独立拟合后才跨尺度比较。','']
    for number,detector,title in [(11,'small','Small-only'),(12,'medium','Medium-only'),(13,'large','Large-only'),(14,'multiscale','MultiScale')]:
        view=selected.loc[selected.detector.eq(detector),['budget','actual_coverage','overall_recall','CHIN_recall','OPCID_recall','CHID_recall','region_count']]
        lines += [f'## {number}. {title} 5/10/20/30% Recall','',_table(view),'']
    lines += ['## 15. CHIN Recall','',
        '四个 detector 的 CHIN 结果见上述统一 coverage 表及 `multiscale_coverage_recall.csv`。','',
        '## 16. OPCID Recall','',
        '四个 detector 的 OPCID 结果使用相同实际 genomic union coverage，不使用 Top-X windows。','',
        '## 17. CHID Recall','',
        '四个 detector 的 CHID 结果同样为 known center 落入 peak-window region union。','',
        '## 18. Coverage-random 95%范围','',_table(random_cov),'',
        '## 19. Region-random 95%范围','',_table(random_reg),'',
        f"两种 baseline 均为 {protocol['random_repetitions']} 次；区间是随机重复的2.5/97.5分位，不是均值置信区间。Region random 保留正式 MultiScale region 的数量和长度多重集。",'',
        '## 20. MultiScale 在10%是否稳定超过random？','',
        f"`{readiness['checks']['beats_coverage_random_at_10pct']}`。",'',
        '## 21. MultiScale 在20%是否稳定超过random？','',
        f"`{readiness['checks']['beats_coverage_random_at_20pct']}`。",'',
        '## 22. 哪些 CHIN 被 Small rescue？','',
        '以下为20% coverage下 Small-only 的 CHIN；其他预算和逐尺度 membership 见 `known_structure_scale_audit.csv`。','',
        _table(audit.loc[audit.type.eq('CHIN')&audit.rescue_category_at_20pct.eq('small_only'),
                         ['structure_id','start','end','small_best_or','small_best_rank']]),'',
        '数量汇总：','',
        _table(rescue.loc[rescue.type.eq('CHIN')]),'',
        '## 23. 哪些 CHID 被 Large rescue？','',
        '以下为20% coverage下 Large-only 的 CHID；逐结构三个 scale 的分数、排名和 membership 见 audit。','',
        _table(audit.loc[audit.type.eq('CHID')&audit.rescue_category_at_20pct.eq('large_only'),
                         ['structure_id','start','end','large_best_or','large_best_rank']]),'',
        '数量汇总：','',
        _table(rescue.loc[rescue.type.eq('CHID')]),'',
        '## 24. 多少结构仍 missed by all scales？','',
        _table(rescue[['type','budget','n_total','missed_all_scales']]),'',
        '## 25. MultiScale candidate region 数量','',f"{statistics['n_candidate_regions']} 个排序 peak-window regions。",'',
        '## 26. Rep1/Rep2 consistency','',json.dumps(statistics['replicate_consistency'],ensure_ascii=False),'',
        '## 27. Zero-axis enrichment','',json.dumps(statistics['zero_axis'],ensure_ascii=False),'',
        '## 28. Readiness 每一项','',
        _table(pd.DataFrame([{'check':k,'passed':v} for k,v in readiness['checks'].items()])), '',
        '## 29. ready_for_task2b','',f"`{str(readiness['ready_for_task2b']).lower()}`。Task2B executed=`{readiness['task2b_executed']}`。",'',
        '## 30. 根因诊断（仅解释，不调参）','',statistics['diagnosis'],'',
        '## 科学和实现限制','',
        '- Expected 在每个 replicate 上仅计算一次到 distance127，各尺度严格切片。',
        '- Formal coverage 使用 peak window；support union 只作诊断，Large support 不会扩张正式预算。',
        '- Genome scan 与 random 均为线性坐标，未引入 circular wrap-around。',
        '- `best_scale`/rescue 是冻结后的解释，不参与尺度选择或权重学习。',
        '- 即使 readiness=true，本流水线也不会自动进入 Task2B。','']
    paths.summary_path.write_text('\n'.join(lines),encoding='utf-8-sig')
