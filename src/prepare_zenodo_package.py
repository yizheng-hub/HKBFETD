"""Prepare derived products and reproducibility metadata; no remote writes."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ai_cache import file_hash
from run_pipeline import RELEASE_FIGURES
from config import CSDI_METADATA_PATH, OUTPUT_DIR, INTERMEDIATE_DIR

out=Path(OUTPUT_DIR)
acceptance=json.loads((out/'final_acceptance.json').read_text(encoding='utf-8'))
if not (acceptance.get('status')=='ACCEPTED_FOR_RELEASE' and
        acceptance.get('engineering_gates_passed') and acceptance.get('fresh_independent_reviews_passed') and
        acceptance.get('dataset_sha256')==file_hash(out/'HKBFETD_v2.parquet')):
    raise RuntimeError('Current dataset has not passed the recorded engineering and fresh independent acceptance gates')
destination=ROOT/'output/zenodo_upload_v2'
marker=destination/'HKBFETD_PACKAGE.txt'
if destination.exists() and not marker.exists():
    raise FileExistsError('Existing directory is not this task\'s package: '+str(destination))
destination.mkdir(exist_ok=True)
marker.write_text('HKBFETD v2 derived-data publication candidate. No upload or publication performed.\n',encoding='utf-8')
files=['HKBFETD_v2.csv','HKBFETD_v2.geojson','HKBFETD_v2.parquet','RELEASE_NOTES_zh.txt',
    'field_dictionary.csv','release_checksums.json','release_validation.json','dimension_qa_summary.json',
    'comparison_summary.json','component_revision_validation.json','height_only_fixed_model_sensitivity.json',
    'model_validation.json','pipeline_run_manifest.json','environment_versions.json','requirements.lock.txt',
    'conda-win64-explicit.txt','unit_test_results.txt','legacy_preservation_validation.json','v2_source_snapshot.zip']
files += ['reproduction_inputs.zip']
files += [name for name in ['attribute_enrichment_summary.json','attribute_enrichment_field_changes.csv',
    'documented_attribute_evidence.csv','attribute_source_evidence.zip','enrichment_independent_review.txt',
    'ROW_REVIEW_STATUS_zh.txt','row_review_coverage.json','row_revision_validation.json',
    'row_revision_field_changes.csv','documented_function_evidence.csv',
    'native_row_review_evidence.zip','unified_repair_validation.json',
    'downstream_reproduction_validation.json','risk_delta_after_unified_repair.json',
    'final_acceptance.json','component_source_reconciliation.json',
    'source_reconciliation_evidence.zip','revision_20261025_summary.json',
    'landmark_building_reference_attributes.csv','LANDMARK_REFERENCE_README_zh.txt',
    'landmark_source_resolution_evidence.zip'] if (out/name).exists()]
for name in files:
    shutil.copy2(out/name,destination/name)
for name in ['reproduction_bundle_validation.json','release_format_validation.json',
             'clean_source_unit_test_results.txt','clean_source_validation.json',
             'independent_review_report.txt','independent_reviews.zip',
             'code_release_reference.json','github_sync_verification.json',
             'FINAL_REVISION_REPORT_zh.txt']:
    if (out/name).exists():
        shutil.copy2(out/name,destination/name)
changes=['v1_1_to_v2_building_changes.csv','v1_1_to_v2_class_transition.csv',
    'all_common_field_change_summary.csv','all_common_field_changes.parquet',
    'height_coverage_by_structure_type.csv',
    'terrain_baseline_to_component_revision.csv','height_only_fixed_model_sensitivity.csv',
    'small_tall_resolved_structures.csv','verified_nonbuilding_structures.csv','stratified_manual_validation_template.csv',
    'model_spatial_split.csv','model_holdout_predictions.csv','reader_cases_v2.csv']
changes += [name for name in ['before_enrichment_to_v2_attribute_changes.parquet',
    'before_row_review_to_v2_attribute_changes.parquet','before_risk_repair_field_changes.csv',
    'before_risk_repair_all_field_changes.parquet','reviewed_typical_cases.csv',
    'before_source_scope_repair_field_changes.csv','before_source_scope_repair_changes.parquet',
    'source_scope_repair_cases.csv'] if (out/name).exists()]
with zipfile.ZipFile(destination/'comparisons_and_validation.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name in changes:archive.write(out/name,name)
    for name in RELEASE_FIGURES:archive.write(out/'figures'/name,'figures/'+name)
    for path in [ROOT/'data/sources'/name for name in ['source_manifest.json','landsd_dtm_building_points.metadata.json',
            Path(CSDI_METADATA_PATH).name,'osm_structural_objects_20260420.metadata.json']] + [ROOT/'ctl/verified_structure_evidence.csv']:
        # Raw provider datasets are kept in the separate reproduction_inputs.zip.
        archive.write(path,'source_provenance/'+path.name)
with zipfile.ZipFile(destination/'trained_model.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    archive.write(Path(INTERMEDIATE_DIR)/'ml_models.pkl','ml_models.pkl')
validation=json.loads((out/'component_revision_validation.json').read_text(encoding='utf-8'))
metrics=json.loads((out/'model_validation.json').read_text(encoding='utf-8'))
comparison=json.loads((out/'comparison_summary.json').read_text(encoding='utf-8'))
sensitivity=json.loads((out/'height_only_fixed_model_sensitivity.json').read_text(encoding='utf-8'))
quality=json.loads((out/'release_validation.json').read_text(encoding='utf-8'))
permitted=[name for name,details in metrics['deployment_class_gate']['classes'].items() if details['eligible_for_model_inference']]
report=f'''HKBFETD v2 本轮修订与验收

数据融合、维度及面积处理、功能规则、分类模型和复现流程已重新运行。原 341,153 个结构 ID 全数保留；原版数据和 {validation['original_artifacts_preserved']} 份受保护文件的哈希保持一致。
塔楼、裙楼的主高度采用同一结构顶部减底部高程，独立保存地面至顶部估计；缺少可靠底部时不以 15 m、海拔或整栋建筑标签替代。
结构本体高度有值 {validation['component_height_known']:,} 条、明确空值 {validation['component_height_unknown']:,} 条；可用报告层数 {validation['reported_storeys_known']:,} 条。
主用途明确标为未知 {validation['function_unknown']:,} 条，不能将有来源或模型推断的类别解释为全量实测标签。按塔楼、裙楼、临时及开放结构分组的高度、层数和功能覆盖率见 height_coverage_by_structure_type.csv。
按 0.1 m 四舍五入统计，15.0 m 记录从 v1.1 的 {validation['v1_1_height_rounded_to_15m']:,} 条变为 {validation['height_rounded_to_15m']:,} 条；新版这些数值来自结构顶底高程差，不是缺失默认值。
舍弃报告层数的冲突记录 {quality['floor_conflict_records']:,} 条，原报告字段仍保留。高度除以 3 m 仅为另列假设，不进入报告层数和楼面面积。
小占地高结构阈值命中 {validation['small_tall_threshold_records']} 条，均为有官方桥梁资料和空间位置支撑的桥塔；未解释的此类记录 {validation['small_tall_unresolved_records']} 条。桥塔高度保留，不作为楼宇能耗面积。
全部非楼宇结构识别 {quality['nonbuilding_structures_excluded_from_energy_area']} 条；共享/多份许可证 {quality['shared_permit_not_allocated_records']:,} 条不再将组级面积重复赋给各栋。
屋宇署用途与年份需同时通过唯一组件点位匹配和来源完整性门。地址与坐标矛盾或组件身份未决的记录不捐赠属性，原值保留为候选；多点、共享点、近邻点不作为确定属性。名称中的“大厦/Plaza”等不能单独确定办公室功能。
面积证据、POI 权重、模型概率分别存储。单功能比例明确标为假设；缺少分用途面积的混合比例留空。

模型主类空间留出宏 F1={metrics['macro_f1']:.6f}，仅衡量与来源推断弱标签的一致性。每类还需至少 30 条超过置信阈值的留出预测、来源标签精确率至少 0.80，才允许模型补齐。
本次允许模型补齐的主类：{', '.join(permitted) or '无'}。其他模型候选留空；已有有效来源类别保留。子类独立执行同样门槛，详情见 model_validation.json。
主类门槛通过训练分区内部的独立空间组选择，未使用外部留出标签；门槛及禁用状态：{json.dumps(metrics['threshold_calibration']['thresholds_by_predicted_class'],ensure_ascii=False)}。1.01 是禁用自动补齐的标记，不是一个有效概率值。
校准门槛保留的外部留出预测，对来源标签的一致性为 {metrics['calibrated_threshold_holdout']['source_label_agreement']:.4%}，覆盖该留出集 {metrics['calibrated_threshold_holdout']['coverage']:.4%}。这些仅是弱标签评价，不是全数据集的实测准确率。
留出结果用于筛选可发布类别，不能将同一结果当作筛选后的独立实测准确率。
v1.1→v2 主类变化 {comparison['changed']['Main_Class']:,} 条、子类变化 {comparison['changed']['Sub_Class']:,} 条；全部共有字段都有逐项新旧对照。
在同一最终模型和其他特征固定时，替换新旧高度会改变 {sensitivity['argmax_class_changed_records']:,} 条模型最高分类，其中 {sensitivity['eligible_final_function_decision_changed_records']:,} 条影响可补齐用途的决策。此为敏感性分析，不是独立测量的纠错数。

默认流程不调用大语言模型或影像 API。原有多模态结果保留为历史证据，已恢复可用 200 张图像，并以输入/模型/提示/图像哈希支持后续断点续跑。
本轮物理修复和本地功能模型重训不依赖重新下载已丢失图像。未伪造全量多模态复核结果。
完整冻结输入：output/v2/reproduction_inputs.zip；精确代码、依赖、各阶段输入输出哈希和模型随本地版本保存。
韧性项目和手稿未纳入本轮读写范围。此文件夹不含它们的文件，也未执行 Zenodo 上传或发布。

验收边界：程序/结构一致性检查不能证明全港逐栋物理属性零误差。来源缺失、测量时间及用途弱标签的不确定性均显式标记，不作为确定真值发布。
独立智能体复查的发现、修复和最终状态见 independent_review_report.txt。
'''
if (out/'attribute_enrichment_summary.json').exists():
    enrichment=json.loads((out/'attribute_enrichment_summary.json').read_text(encoding='utf-8'))
    report += f"\n补充修订的历史阶段：采用地政总署2026-10-20官方批量快照，保留记录创建/更新时间，不将其冒充竣工或测量日期。历史属性增补报告覆盖{enrichment['web_researched_buildings']}个有名建筑记录（包括排除错配来源的记录），不是全港逐栋网页核实；该阶段统计的层数文档{enrichment['component_storey_documents']}项、GFA文档{enrichment['component_gfa_documents']}项不代表全部仍获当前主值采纳。\n中环广场塔楼75层保留。泓富产业千禧广场的32层曾在历史增补阶段采纳，本轮发现跳号楼层标签与原始25层没有可靠实体层对应，已撤销主层数及依赖代理面积；办公用途文档仍可采用。朗豪坊59个报告层级的物理层数口径未决，主层数与依赖代理面积留空。来源旧值、其他计数口径、地下层及不同面积种类分别保留。文档GFA不冒充登记GFA，住宅单位内部面积及可出租面积不当作GFA。\n历史阶段对照及覆盖变化见attribute_enrichment_summary.json；本轮当前来源采纳与撤值以component_source_reconciliation.json和revision_20261025_summary.json为准。原检查点、模型、输入与代码在本地v2_before_enrichment_20261023及history_20261025归档中保留。\n"
if (out/'ROW_REVIEW_STATUS_zh.txt').exists():
    report = (out/'ROW_REVIEW_STATUS_zh.txt').read_text(encoding='utf-8') + '\n' + report
    report += '\n逐条复查后的批量修订：租户POI不再确定整栋用途，地址及连接天桥描述不再冒充主体用途；房委会产权类别不单独确定住宅用途或公屋子类。模型住房子类仅保留为候选，不自动采用。中环广场文档总高与来源组件高的参考口径尚未调和，主高度留空，原高程与复查前数值保留；75层有组件文档支持。\n'
    if (out/'row_revision_validation.json').exists():
        revision=json.loads((out/'row_revision_validation.json').read_text(encoding='utf-8'))
        report += f"本轮相对90300be候选版：主类改变{revision['changed_main_classes']:,}条，子类改变{revision['changed_subclasses']:,}条，明确绑定组件的用途文档事实{revision['function_facts_checked']}项。此为规则、来源及模型联合修订，不是独立实测纠错数量。\n"
if (out/'revision_20261025_summary.json').exists():
    current=json.loads((out/'revision_20261025_summary.json').read_text(encoding='utf-8'))
    report += f"\n最终来源口径修订：相比紧邻的第9轮候选，主类改变{current['changed_main_class']:,}条、子类改变{current['changed_subclass']:,}条、任一改变{current['changed_either_class']:,}条。它们包含用途来源更新和真实重训的影响，不是高度纠错数量。详见revision_20261025_summary.json及comparisons_and_validation.zip内source_scope_repair_cases.csv。\n中银大厦主高度因303.3m组件高与315m结构高缺少共同基准而未知，65/70/72层分源保留。ICC保留490−29.4=460.6m结构组件高度，484m建筑高度另列；103/108/118层未调和，主层数及依赖面积未知，不能将相邻5层组件相加。KCC一号塔26/35层未调和，同样只保留可靠办公用途，主层数及依赖面积未知。所有原始来源数值、口径和撤值理由见source_reconciliation_evidence.zip。\n"
(out/'FINAL_REVISION_REPORT_zh.txt').write_text(report,encoding='utf-8')
if (out/'LANDMARK_REFERENCE_README_zh.txt').exists():
    report += '\n'+(out/'LANDMARK_REFERENCE_README_zh.txt').read_text(encoding='utf-8')
    (out/'FINAL_REVISION_REPORT_zh.txt').write_text(report,encoding='utf-8')
(destination/'FINAL_REVISION_REPORT_zh.txt').write_text(report,encoding='utf-8')
description=f'''HK-BFETD: Hong Kong Building Function and Energy Typology Dataset — version 2.0

This revision retains the original 341,153 structural records for an ID-complete comparison with the published v1.1.0 dataset (https://doi.org/10.5281/zenodo.20710499). It revises physical-attribute fusion, source-scope controls, functional inference, area/proportion semantics and the reproduction workflow.

The primary height is structural component top elevation minus base elevation, with towers and podiums treated separately. The frozen official 2026-10-20 bulk download of the Lands Department CSDI Building layer is matched by unique component identifier, compatible structure type and valid footprint intersection-over-union of at least 0.95. Ground-to-top estimates and OSM reported heights have separate fields. Heights are not replaced by a default 15 m or by uncorrected datum elevations. Retained record creation/update dates are not survey or construction dates.

Component heights are available for {validation['component_height_known']:,} records; {validation['component_height_unknown']:,} remain explicitly null. Reported, conflict-checked storeys are available for {validation['reported_storeys_known']:,} records. Height-derived storeys do not enter reported counts or GFA. Unknown/conflicting source values are not fabricated. Eight small-footprint/tall objects have been identified as bridge pylons and retained as nonbuilding structural records excluded from building energy-area allocation.

Source-scoped documents reconcile supported component storey counts; unresolved physical-storey references remain null. Alternative counts, basement levels, mechanical/refuge conventions and floor labels remain separate evidence. GFA prefers scoped registration, then documented same-component gross floor area, then an explicit footprint-times-reported-storeys approximation. Owner-reported approximate areas remain approximate. Rentable, saleable and unit internal areas are not GFA. Shared permits are not allocated in full to every structure, and administrative uniqueness alone does not establish physical scope. POI evidence weights, permit accounting splits and classifier probabilities are not measured current-use area fractions. Single-function vectors are documented assumptions; unsupported mixed-use shares remain null.

Function models were retrained using disjoint 2 km spatial blocks and training-partition imputation. Spatial holdout macro-F1 is {metrics['macro_f1']:.6f} against source-inferred weak labels; this is not independent field-survey accuracy. Blocks do not impose a 2 km buffer between every training and holdout building. Unknown source conflicts remain unknown despite high model probability. Main/subclass provenance, source/quality flags, full field definitions, old/new comparisons and exploratory fixed-model height sensitivity accompany the products.

Model-filled main/subclasses also require at least 30 confidence-qualified spatial-holdout predictions and precision of at least 0.80 against source labels. A class failing this deployment gate remains unknown. The same holdout is used to screen deployable classes, so its reported scores are not independent accuracy after selecting classes.
Main-class score cutoffs are selected using a separate spatial calibration partition inside the outer training data, requiring at least 30 predictions and precision of at least 0.90 against source labels. Outer holdout labels do not select these cutoffs; model probabilities themselves remain uncalibrated.

The inventory is the original 2025-02-06 footprint stock supplemented by frozen supporting evidence; it is not a claim of complete present-day building coverage. The code, trained model and reproduction_inputs.zip are included. Input/source/output hashes and locked environment information document the exact local reference run. Exact replay uses the frozen inputs; the default workflow makes no external AI or image requests. Supplementary source evidence distinguishes source coverage and component-match candidates from primary dataset attributes; online research coverage is explicitly reported rather than described as a whole-stock web audit.

Create a new version of the existing Zenodo record and retain its creator and rights metadata. This file is a description draft, not a newly assigned DOI or a claim of zero physical errors. Software licensing does not replace the dataset/source-provider terms.
'''
if (out/'ROW_REVIEW_STATUS_zh.txt').exists():
    description = 'Review scope: whole-inventory engineering checks and the agreed Buildings Department related risk review. This does not certify every attribute as independently surveyed. See ROW_REVIEW_STATUS_zh.txt and row_review_coverage.json.\n\n' + description
    description += '\nResidential tenure subclasses are never filled solely by the model. Tenant POIs, addresses and access-link descriptions do not establish whole-component use. Documented component facts and prior decisions are retained. Known unresolved height-reference differences withhold the primary height rather than forcing architectural total height into a structural component field.\n'
(destination/'ZENODO_DESCRIPTION_EN.txt').write_text(description,encoding='utf-8')
(destination/'READ_ME_FIRST_zh.txt').write_text('''HKBFETD v2 上传候选包
主高度采用结构本体顶部减底部高程，地面至顶部另列。首先阅读 RELEASE_NOTES_zh.txt 和 field_dictionary.csv。
CSV、GeoJSON、GeoParquet 为同一版本的三种格式，均保留原 341,153 个建筑/结构 ID；推荐 GIS 使用 GeoParquet。
comparisons_and_validation.zip 包括逐栋新旧对照、空间验证、桥塔处理结果、独立验证抽样模板及来源元数据。
trained_model.zip 为本次重训模型；v2_source_snapshot.zip 为确切源代码、控制文件、测试及环境信息。
reproduction_inputs.zip 保存本次冻结输入；在新的源码目录中解压并保留相对路径，再运行 python src/run_pipeline.py --run-root reproduction_run --input-bundle，将复现输出写入独立目录。
原始输入作为单独压缩包随候选包保存；各来源元数据和提供方条款仍适用。
在现有 https://zenodo.org/records/20710499 中创建新版本，保留作者及数据许可信息。新版本 DOI 由 Zenodo 发布时生成。
独立智能体的程序/数据检查结果完成后列于 independent_review_report.txt。这类检查不等同于对全香港逐栋实测，也不构成绝对零误差保证。
''',encoding='utf-8')
if (out/'ROW_REVIEW_STATUS_zh.txt').exists():
    first=destination/'READ_ME_FIRST_zh.txt'
    first.write_text('【已按约定风险范围验收】全库工程检查与屋宇署关联风险核查的范围、来源及未证实项见ROW_REVIEW_STATUS_zh.txt；不宣称全库每个属性均经独立实测。\n\n'+first.read_text(encoding='utf-8'),encoding='utf-8')
if (out/'LANDMARK_REFERENCE_README_zh.txt').exists():
    first=destination/'READ_ME_FIRST_zh.txt'
    first.write_text('【中银/ICC整栋属性补充】landmark_building_reference_attributes.csv明确保存整栋口径的已查明数值及来源；请先读LANDMARK_REFERENCE_README_zh.txt，区别整栋与组件主字段，不将整栋层数自动用于组件面积。\n\n'+first.read_text(encoding='utf-8'),encoding='utf-8')
inventory={p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(destination.iterdir()) if p.is_file() and p.name not in ['SHA256SUMS.txt','package_manifest.json']}
manifest={'version':'2.0','created_utc':datetime.now(timezone.utc).isoformat(),
    'previous_record':'https://zenodo.org/records/20710499','remote_actions':'none',
    'frozen_inputs_archive':'reproduction_inputs.zip','files':inventory}
if (out/'ROW_REVIEW_STATUS_zh.txt').exists():
    manifest['row_review_status']='ACCEPTED_SCOPED_REVIEW'
manifest['final_acceptance']=acceptance
(destination/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
inventory['package_manifest.json']={'sha256':file_hash(destination/'package_manifest.json')}
(destination/'SHA256SUMS.txt').write_text(''.join(record['sha256']+'  '+name+'\n' for name,record in sorted(inventory.items())),encoding='utf-8')
print(json.dumps({'package':str(destination),'files':len(inventory),'bytes':sum(p.stat().st_size for p in destination.iterdir() if p.is_file())},ensure_ascii=False),flush=True)
