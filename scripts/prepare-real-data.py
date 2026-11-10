"""Fetch official UCI archives and prepare audited, chronological ML tasks.

Optional Excel dependency: openpyxl>=3.1,<4 (dataset preparation only).
Do not put raw or derived customer data in version control.
"""
import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--extra-deps", type=Path)
    args = parser.parse_args()
    if args.extra_deps:
        sys.path.insert(0, str(args.extra_deps.resolve()))
    import pandas as pd

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = {
        "retail": "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip",
        "bike": "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip",
    }
    provenance = {}
    for name, url in sources.items():
        archive = root / (name + ".zip")
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=180) as response, archive.with_suffix(".part").open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            archive.with_suffix(".part").replace(archive)
        provenance[name] = {"url": url, "archive_sha256": digest(archive), "license": "CC BY 4.0"}
        print(json.dumps({"downloaded": name, "bytes": archive.stat().st_size}), flush=True)

    with zipfile.ZipFile(root / "retail.zip") as archive:
        # XLSX is itself a ZIP; materialize the inner file for reliable random seeks.
        raw = pd.read_excel(io.BytesIO(archive.read("Online Retail.xlsx")), engine="openpyxl")
    audit = {"raw_rows": len(raw)}
    frame = raw.copy()
    for reason, condition in [
        ("missing_customer", lambda x: x.CustomerID.isna()),
        ("cancelled", lambda x: x.InvoiceNo.astype(str).str.upper().str.startswith("C")),
        ("nonpositive_quantity_or_price", lambda x: (x.Quantity <= 0) | (x.UnitPrice <= 0)),
        ("exact_duplicates", lambda x: x.duplicated()),
    ]:
        mask = condition(frame)
        audit[reason + "_removed"] = int(mask.sum())
        frame = frame.loc[~mask].copy()
    frame["InvoiceDate"] = pd.to_datetime(frame.InvoiceDate)
    frame["CustomerID"] = frame.CustomerID.astype(int)
    frame["spend"] = frame.Quantity * frame.UnitPrice
    audit["clean_rows"] = len(frame)
    audit["unique_customers"] = int(frame.CustomerID.nunique())
    snapshots, windows = [], []
    for anchor in pd.date_range("2011-03-01", "2011-11-01", freq="MS"):
        end = anchor + pd.offsets.MonthBegin(1)
        history = frame.loc[(frame.InvoiceDate >= anchor - pd.Timedelta(days=90)) & (frame.InvoiceDate < anchor)]
        future = frame.loc[(frame.InvoiceDate >= anchor) & (frame.InvoiceDate < end)]
        grouped = history.groupby("CustomerID", sort=True)
        features = grouped.agg(last_purchase=("InvoiceDate", "max"), invoices=("InvoiceNo", "nunique"),
                               spend=("spend", "sum"), quantity=("Quantity", "sum"), products=("StockCode", "nunique"))
        features["recency_days"] = (anchor - features.pop("last_purchase")).dt.total_seconds() / 86400
        features["average_basket"] = features.spend / features.invoices
        features["anchor"] = anchor.strftime("%Y-%m-%d")
        features["target"] = features.index.isin(future.CustomerID).astype(int)
        features.index.name = "customer_id"
        snapshots.append(features.reset_index())
        assert history.InvoiceDate.max() < anchor <= future.InvoiceDate.min()
        assert future.InvoiceDate.max() < end <= frame.InvoiceDate.max()
        windows.append({"anchor": str(anchor.date()), "label_end_exclusive": str(end.date()),
                        "feature_max_time": history.InvoiceDate.max().isoformat(),
                        "rows": len(features), "positive_rate": float(features.target.mean())})
    retail = pd.concat(snapshots, ignore_index=True)
    retail.to_csv(root / "retail.csv", index=False, float_format="%.12g", lineterminator="\n")
    provenance["retail"].update(audit=audit, windows=windows, derived_rows=len(retail),
                               derived_sha256=digest(root / "retail.csv"), features=[c for c in retail if c not in {"customer_id", "anchor", "target"}])

    with zipfile.ZipFile(root / "bike.zip") as archive:
        with archive.open("hour.csv") as stream:
            bike = pd.read_csv(stream)
    assert (bike.casual + bike.registered == bike.cnt).all()
    raw_count = len(bike)
    bike["timestamp"] = (pd.to_datetime(bike.dteday) + pd.to_timedelta(bike.hr, unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%S")
    bike = bike.drop(columns=["instant", "dteday", "casual", "registered"]).rename(columns={"cnt": "target"})
    # Preserve documented categorical semantics; ordinal numerical fields remain numeric.
    for col in ["season", "mnth", "hr", "weekday", "weathersit"]:
        bike[col] = "category_" + bike[col].astype(str)
    assert bike.timestamp.is_unique
    bike.to_csv(root / "bike.csv", index=False, lineterminator="\n")
    provenance["bike"].update(raw_rows=raw_count, derived_rows=len(bike), excluded_target_components=["casual", "registered"],
                             derived_sha256=digest(root / "bike.csv"), features=[c for c in bike if c not in {"timestamp", "target"}])
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"retail_rows": len(retail), "bike_rows": len(bike), "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
