"""Offline numerical report with safe HTML and inline SVG charts."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def write_prediction_index(path: Path, result: dict) -> None:
    rows = []
    for item in result["items"]:
        prefix = Path(item["file"]).with_suffix("").as_posix()
        overlay, mask = quote(prefix + "_overlay.png"), quote(prefix + "_mask.png")
        rows.append(f'<article><h2>{escape(item["file"])}</h2><p>{item["width"]}×{item["height"]} · 缺陷像素 {item["defect_area_px"]} · 模型得分 {item["image_score"]:.3f} · 整图判断 {"疑似缺陷" if item["predicted_defect"] else "未触发阈值"}</p><a href="{overlay}"><img loading="lazy" src="{overlay}" alt="预测叠加图"></a><p><a href="{mask}">二值掩码</a> · 区域数 {len(item["regions"])}</p></article>')
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>批量缺陷预测</title><style>body{font:17px/1.7 system-ui,"Microsoft YaHei";max-width:1000px;margin:35px auto;padding:0 20px;background:#f5f8fc;color:#203047}article{background:white;border-radius:14px;padding:20px;margin:20px 0}img{max-width:100%;max-height:500px}h2{overflow-wrap:anywhere}</style><h1>批量缺陷预测</h1><p>共 ' + str(result["count"]) + ' 张；红色为预测缺陷。未提供人工真值，不能由这些图片得出精度结论。模型得分未经概率校准。</p><p><a href="predictions.json">JSON：每个区域的面积和边框</a></p>' + ''.join(rows) + '</html>'
    path.write_text(text, encoding="utf-8")


def write_report(path: Path, report: dict, title: str = "磁瓦缺陷分割评估") -> None:
    detection = report.get("detection", {})
    confusion = detection.get("confusion", {})
    category_rows = "".join(
        f'<tr><td>{escape(name)}</td><td>{row["count"]}</td><td>{row["pixel_micro_dice"]:.3f}</td><td>{row["pixel_micro_iou"]:.3f}</td></tr>'
        for name, row in report.get("per_category", {}).items()
    )
    curve = detection.get("precision_recall_curve", [])
    points = " ".join(f'{45 + point["recall"] * 500:.1f},{235 - point["precision"] * 200:.1f}' for point in curve)
    interval = report.get("dice_bootstrap_95_ci")
    interval_text = f'{interval["low"]:.3f} - {interval["high"]:.3f}（按图片重采样的 95% 区间）' if interval else "未计算"
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>body{{font:17px/1.7 system-ui,'Microsoft YaHei';max-width:980px;margin:35px auto;padding:0 24px;color:#203047;background:#f5f8fc}}section{{background:white;padding:22px;border-radius:14px;margin:18px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d9e3ec;text-align:left;padding:9px}}.score{{font-size:30px;color:#176c86}}svg{{width:100%;max-width:600px}}code{{overflow-wrap:anywhere}}</style>
<h1>{escape(title)}</h1><p>同一数据集固定留出测试；参数由验证集选择。此报告未包含数据集图像。</p>
<section><h2>像素分割</h2><div class="score">Dice {report.get("pixel_micro_dice", 0):.3f} · IoU {report.get("pixel_micro_iou", 0):.3f}</div>
<p>测试图片 {report.get("count", 0)} 张；像素 TP/FP/FN = {report.get("pixel_tp", 0)} / {report.get("pixel_fp", 0)} / {report.get("pixel_fn", 0)}。</p>
<p>Dice 衡量缺陷区域重叠，不是图片识别准确率。置信区间：{interval_text}</p></section>
<section><h2>整图有无缺陷</h2><p>TP={confusion.get("tp", "—")}，FP={confusion.get("fp", "—")}，FN={confusion.get("fn", "—")}，TN={confusion.get("tn", "—")}。</p>
<p>F1={report.get("image_f1", 0):.3f} · AUROC={report.get("image_auroc", 0) or 0:.3f} · AP={detection.get("average_precision", 0) or 0:.3f} · 正常样本误报率={detection.get("false_positive_rate", 0) or 0:.1%}</p>
<svg viewBox="0 0 600 280" role="img" aria-label="图像检测PR曲线"><path d="M45 30V235H555" fill="none" stroke="#789"/><polyline points="{points}" fill="none" stroke="#008e85" stroke-width="3"/><text x="250" y="270">召回率 Recall →</text><text x="45" y="20">精确率 Precision ↑</text></svg></section>
<section><h2>逐类结果</h2><table><tr><th>类别</th><th>张数</th><th>Dice</th><th>IoU</th></tr>{category_rows}</table></section>
<section><h2>复查与边界</h2><p>优先检查漏检和误报。小缺陷、类别不平衡、采集环境变化都会影响表现；本结果不等同于生产线验证。</p><details><summary>模型与复现信息</summary><pre>{escape(json.dumps(report.get("experiment", {}), ensure_ascii=False, indent=2))}</pre></details></section></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
