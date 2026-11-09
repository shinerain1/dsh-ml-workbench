"""Portable, dependency-free HTML report with escaped user-controlled labels."""

from html import escape


def html_report(record: dict, result: dict, dataset: dict) -> str:
    def table(headers, rows):
        head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
        body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    metrics = table(["Metric", "Model", "Dummy baseline"], [
        [key, f"{value:.8g}", f"{result['baseline_metrics'][key]:.8g}" if key in result.get("baseline_metrics", {}) else "—"]
        for key, value in result["metrics"].items()])
    confusion = ""
    if "confusion_matrix" in result:
        classes = result["classes"]
        confusion = "<h2>Confusion matrix</h2><p>Rows: actual. Columns: predicted.</p>" + table(
            ["Actual / predicted", *classes], [[label, *row] for label, row in zip(classes, result["confusion_matrix"], strict=True)])
    split = dataset.get("split", {"strategy": "random (legacy run)"})
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Experiment {escape(record['name'])}</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#172b3a;background:#f5f8fa;max-width:980px;margin:40px auto;padding:0 24px}}
h1,h2{{color:#135a54}}table{{border-collapse:collapse;background:white;width:100%;margin:20px 0}}
th,td{{text-align:left;border-bottom:1px solid #d8e2e7;padding:10px}}code{{overflow-wrap:anywhere}}.note{{padding:16px;background:#e8f3ef}}
</style><h1>{escape(record['name'])}</h1><p>Model: <b>{escape(record['model'])}</b> · Task: {escape(record['task'])}</p>
<p>Run: <code>{escape(record['run_id'])}</code></p><p>Parent: <code>{escape(str(record['parent_run_id']))}</code></p>
<h2>Validation metrics</h2>{metrics}{confusion}<h2>Evaluation boundary</h2>
{table(['Property', 'Value'], list(split.items()))}<p>Data SHA-256: <code>{escape(dataset['sha256'])}</code></p>
<p>Evaluation key: <code>{escape(record['comparison_key'])}</code></p>
<p>Fit time: {result['train_seconds']:.6f} seconds (one local measurement).</p>
<p class="note">Validation scores are not independent final-test estimates after repeated model selection.
Check whether the data and split reflect your deployment scenario. A dummy baseline is a sanity check, not proof of business value.</p>
<h2>Reproduce</h2><p>Install <code>inputs/requirements.txt</code>, then run <code>python reproduce.py</code>.
If the CSV is omitted, add <code>--data ORIGINAL.csv</code>. See REPORT.md.</p></html>'''
