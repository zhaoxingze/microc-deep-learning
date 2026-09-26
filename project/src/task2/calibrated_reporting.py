"""Task 2A.2 reports distinguish validation evidence from diagnostic hypotheses."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.task2.reporting import plot_training_curve


def _table(frame):
    def fmt(value):
        if isinstance(value,(float,np.floating)):
            return f'{value:.6f}' if np.isfinite(value) else 'NA'
        return str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def write_calibrated_report(paths,config,protocol,statistics,tables,scores,history):
    root=paths.output_root
    curve=tables['coverage_recall_curve'];audit=tables['known_structure_detection_audit']
    budgets=curve.loc[np.isclose(curve.budget.to_numpy()[:,None],[.05,.10,.20,.30]).any(axis=1)]
    fig,ax=plt.subplots(figsize=(8,5))
    for label in ('overall','CHIN','OPCID','CHID'):
        ax.plot(curve.actual_coverage*100,curve[f'{label}_recall']*100,label=label)
    ax.set(xlabel='Actual genomic union coverage (%)',ylabel='Known-center recall (%)',
           title=f'Calibrated OR: region-first ({config.mode})',xlim=(0,max(40,float(curve.actual_coverage.max()*100))),ylim=(0,100))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(root/'coverage_recall_curve.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,branch in zip(axes,('density','shape','or')):
        source='calibrated_or_score' if branch=='or' else f'{branch}_tail_score'
        field='best_calibrated_or_score' if branch=='or' else f'best_{branch}_tail_score'
        data=[audit.loc[audit.type.eq(c),field].dropna().to_numpy() for c in ('CHIN','OPCID','CHID')]
        data.append(scores.loc[scores.background_candidate,source].dropna().to_numpy())
        # ECDF handles missing smoke classes and saturated empirical tails explicitly.
        for label,values in zip(('CHIN','OPCID','CHID','Background'),data):
            if len(values): ax.step(np.sort(values),np.arange(1,len(values)+1)/len(values),where='post',label=f'{label} (n={len(values)})')
        ax.set(title=f'{branch.title()} anomaly',xlabel='Anomaly score',ylabel='Empirical CDF');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Known: best OR overlapping window per structure; background: eligible scan windows',fontsize=10)
    fig.tight_layout();fig.savefig(root/'known_score_distributions.png',dpi=160);plt.close(fig)
    plot_training_curve(history,root/'training_curve.png')
    corr=statistics['correlations'];ready=statistics['readiness']
    lines=['# Task 2A.2 calibrated detector summary','',
        '> Smoke：仅检查流程，不支持科学结论。' if config.mode=='smoke' else '> Full：一次固定协议运行；已知标注仅用于背景排除和最终评价，未用于参数选择。','',
        '## 1. 为什么 robust z-score 不足？','',
        '相同 robust z 不代表相同背景尾部罕见程度，两个分支的尾部形状可能不同。','',
        '## 2. 为什么 empirical tail？','',
        '用独立背景经验上尾转换到同一罕见度量纲；这是背景相对 anomaly，不是已证明的生物学显著性。重叠窗口具有空间依赖，不能当成独立检验。','',
        '## 3. Calibration set 如何构建？','',
        f"背景窗口数：{statistics['background_counts']}；完整 block 数：{statistics['block_counts']}。",
        '预先固定 seed 的整块三分，目标 70/15/15，实际窗口比例允许偏离；跨块窗口和已知结构排除区域不进入三分。',
        f"实际背景窗口比例：{protocol.get('actual_background_fractions',{})}。",
        'Train 拟合 clip 并训练 AE；Validation 只选 checkpoint；Calibration 只拟合分支尾分布。',
        f"AE best epoch={statistics['best_epoch']}；best validation loss={statistics['best_validation_loss']:.9f}。",'',
        '## 4. 是否无 overlap？','',
        '执行独立 Train/Validation/Calibration 两两半开区间重叠检查；任意泄漏立即失败。最终七项检查见下表。','',
        f"边界限制：继承的已知排除是 center±6400，而不是扩展完整标注区间；较长标注仍可能与部分 Train/Val 窗口相交。实际计数：{statistics.get('known_interval_overlap_counts',{})}。详见 background_known_interval_overlap_audit.csv。Calibration 与真实已知区间另行检查，任何相交直接中止，不重新抽划分。",'',
        '## 5. Density tail 如何计算？','',
        '每个 replicate 的 d 为 genome O/E 上三角、距离主对角线 >2 bins 的均值。p_D=(1+count(D_cal>=d))/(n+1)，A_D=-log10(p_D)。','',
        '## 6. Shape tail 如何计算？','',
        '每个 replicate 的 s 为 shape-normalized AE 重构在同一 off-diagonal mask 上的 MSE。p_S=(1+count(S_cal>=s))/(n+1)，A_S=-log10(p_S)。',
        f"最小可达 p={statistics['minimum_empirical_p']:.9f}；更极端值会饱和并产生同分。",'',
        '## 7. Calibrated OR 如何计算？','',
        '每个分支先平均 rep1/rep2 anomaly，再取两个 paired branch 的最大值。min replicate 只作诊断。每重复 Bonferroni 字段不参与排名；平均 -log10(p) 不宣称是有效 combined p-value。',
        f"主分数有 {statistics['calibrated_or_unique_scores']} 个不同值，最高分有 {statistics['calibrated_or_max_ties']} 个窗口并列；同分用 SHA256(seed|window_id) 固定排序，不读取标注。",'',
        '## 8. 为什么不再以 Top-X% windows 为主？','',
        '6400 bp 窗口每 800 bp 滑动，高度重叠。窗口比例不等于实际覆盖预算。主比较固定为 region union coverage；旧窗口比例仅为辅助对照。','',
        '## 9. NMS / region 如何工作？','',
        '降序遍历，小于 3200 bp 的同染色体窗口归属最高排名邻近峰；支持窗口只保留包含峰的连续 800 bp center 分量。区域取成员区间边界，不跨峰链式合并。区域可重叠，覆盖只计并集。',
        '整块加入排序区域至达到预算，保留并报告 overshoot；mixed 标签固定为两分支 anomaly 均 >= -log10(.05)，不影响排序。','',
        '## 10. 真实候选 regions 数量？','',
        f"全排序区域共 {statistics['n_candidate_regions']} 个，来源于 {statistics['n_windows']} 个配对窗口。这不是全部需要后续接受的候选：正式候选集合由固定覆盖预算截取。",'',
        '## 11. 5/10/20/30% coverage Overall Recall','',
        _table(budgets[['budget','actual_coverage','region_count','overall_recall','CHIN_recall','OPCID_recall','CHID_recall']]),'',
        '## 12. CHIN Recall / 13. OPCID Recall / 14. CHID Recall','',
        '逐类主召回见上表，均以 unique structure center 落入候选并集定义；区间 overlap 辅助召回见 coverage_recall_curve.csv。','',
        '## 15. Coverage-matched random','',
        _table(tables['coverage_random_baseline']), '',
        '## 16. Region-matched random','',
        _table(tables['region_random_baseline']), '',
        f"以上区间为 {protocol.get('random_repetitions',100)} 次随机重复的 2.5/97.5 分位范围，不是均值的置信区间。每次抽样覆盖、是否匹配和召回见 random_baseline_draws.csv。Region-random 保持数量及长度多重集，但随机重叠会改变实际并集覆盖。全部使用与 scan 一致的线性域。",'',
        '## 17. 是否稳定优于 random？','',
        f"10% coverage 判定：{ready['checks'].get('beats_coverage_random_at_10pct',False)}；20% coverage 判定：{ready['checks'].get('beats_coverage_random_at_20pct',False)}。两者均超过 coverage-random 上分位才满足稳定富集门槛。",'',
        '## 18. Replicate consistency 与 branch correlation','',
        f"Rep1/Rep2 calibrated OR：{corr['replicates']}。Density/Shape paired anomaly：{corr['branches']}。",'',
        '## 19. Zero-axis enrichment','',
        json.dumps(statistics['zero_axis'],ensure_ascii=False),
        '固定统计 20% 预算区域并集包含中心的扫描窗口，与全部扫描窗口比较；两个 replicate 任一 zero-axis 即记为 zero-axis。','',
        '## 20. 哪些结构仍漏检？','',
        '逐条 ID、最佳分数、排名、最近扫描中心距离、覆盖判定见 known_structure_detection_audit.csv；下表分预算统计。','',
        _table(tables['missed_known_summary']), '',
        '## 21. CHIN / CHID 漏检的分数模式','']
    for label in ('CHIN','CHID'):
        missed=audit.loc[audit.type.eq(label)&~audit.selected_at_20pct_coverage]
        valid=missed.dropna(subset=['best_calibrated_or_score'])
        if len(valid):
            low=((valid.best_density_tail_score < -np.log10(.05))&(valid.best_shape_tail_score < -np.log10(.05))).sum()
            lines.append(f"- {label}：20% coverage 漏检 {len(missed)} 个；最佳 OR 窗口 density/shape/OR 中位数分别为 "
                f"{valid.best_density_tail_score.median():.4f}/{valid.best_shape_tail_score.median():.4f}/{valid.best_calibrated_or_score.median():.4f}；"
                f"两分支均低于固定 p=.05 标尺的有 {low} 个；全区域集合无中心覆盖的有 {missed.first_covering_region_rank.isna().sum()} 个。")
        else:lines.append(f'- {label}：没有可分析的漏检扫描窗口。')
    lines += ['', '高分但未在预算内召回，需结合 first_covering_region_rank/coverage 检查区域优先级或窗口边界；低分提示当前窗口/表征没有把该结构转为显著背景偏离。仅凭这些诊断不能断言具体生物机制或唯一原因。本轮不据此修改 detector。', '',
        '## 22. ready_for_task2b','',f"`{str(ready['ready_for_task2b']).lower()}`",'',
        '## 23. 为什么？全部固定检查','',
        _table(pd.DataFrame([{'check':k,'passed':v} for k,v in ready['checks'].items()])), '',
        '全部门槛同时满足才可冻结该版本为后续候选输入；任何失败均保留 false，并停止 Task2A。无论结果如何，本次没有运行 Task2B、聚类、embedding 或生物学解释。', '',
        '## 对照、限制与可追溯性','',
        '- detector_comparison.csv 的七个方法共用本次 AE 与三分背景；它们是本轮配对消融，不冒充旧模型原始结果。旧输出目录保留了各自真实历史结果。',
        '- known_score_distributions.png：每个 known structure 使用最佳 OR 重叠窗口，背景使用 eligible windows；粒度不同，图用于诊断，不能当成无偏检验。',
        '- Calibration 有限且窗口空间相关，极端分数分辨率受限。SHA tie-break 仅保证确定性，不消除统计分辨率限制。',
        '- MG1655 的 circular boundary handling 尚未启用；scan 与 random 都采用线性坐标，未来可单独改进。',
        '- frozen_protocol.json 保存预先规则、代码哈希与 split 哈希；calibration.json 保存每个重复/分支完整校准分布。Manifest 验证所有输入和输出 SHA256。',
        '- protected_artifacts_audit.json 验证 legacy/refined 结果与原始五个 Task2 数据工件未被覆盖。','']
    paths.summary_path.write_text('\n'.join(lines),encoding='utf-8-sig')
