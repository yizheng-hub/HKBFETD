import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.native_review_progress import GROUPS, binding, export_batch, import_legacy, ingest, status, validate_results


def result(bid, bind, retrieval="opened_this_pass", mode="source_research"):
    no_web = mode == "attribute_consistency_only_no_web"
    item = {"building_id": bid, "record_binding_sha256": bind, "model": "gpt-6-luna",
            "outcome": "partially_supported", "findings": [{"field": "function"}],
            "unverified_groups": ["dimensions"], "reviewed_field_groups": sorted(GROUPS),
            "review_mode": mode, "queries_used": [] if no_web else ["official building information"],
            "sources": [{"url": "https://example.org/source", "retrieval_status": retrieval}]}
    if no_web:
        item["query_not_run_reason"] = "No reliable name or address supports a useful external search."
    return item


class NativeReviewProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "tiny.parquet"
        self.state = self.root / "state"
        self.write_dataset(["a", "b", "c"])

    def tearDown(self):
        self.tmp.cleanup()

    def write_dataset(self, ids, values=None):
        pd.DataFrame({"Building_ID_建筑ID": ids, "public_value": values or list(range(len(ids))),
                      "LandsD_Name_地政署名称": ["香港建築", None, "Tower 3"][:len(ids)],
                      "height_m": [12.5, None, 88.0][:len(ids)]}).to_parquet(self.dataset)

    def test_export_ingest_and_status_only_counts_valid_results(self):
        exported = export_batch(self.dataset, self.state, "batch1", 2)
        rows = [json.loads(x) for x in exported.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        packet = json.loads((self.state / "batch1.compact.json").read_text(encoding="utf-8"))
        restored = []
        for values in packet["public_rows"]:
            restored.append({**packet["public_constants"], **dict(zip(packet["public_columns"], values))})
        self.assertEqual(restored, [{k: v for k, v in row.items() if k != "_review"} for row in rows])
        self.assertEqual(packet["ids"], [row["Building_ID_建筑ID"] for row in rows])
        self.assertNotIn("record_binding_sha256", packet["geometry_constants"])
        self.assertNotIn("content_fingerprint_sha256", packet["geometry_constants"])
        self.assertEqual(status(self.state, self.dataset)["current_dataset"]["reviewed"], 0)
        results = [result(r["Building_ID_建筑ID"], r["_review"]["record_binding_sha256"],
                          "not_retrieved" if i == 0 else "historical_reference",
                          "source_research" if i == 0 else "attribute_consistency_only_no_web")
                   for i, r in enumerate(rows)]
        rp = self.root / "results.json"; rp.write_text(json.dumps(results), encoding="utf-8")
        self.assertEqual(ingest(self.state, "batch1", rp), 2)
        self.assertEqual(status(self.state, self.dataset)["current_dataset"]["reviewed"], 2)
        progress = status(self.state)
        self.assertEqual(progress["fully_fact_verified_records"], 0)
        self.assertEqual(progress["review_modes"], {"source_research": 1, "attribute_consistency_only_no_web": 1})
        self.assertEqual(progress["records_with_some_opened_sources"], 0)

    def test_rejects_binding_duplicate_and_incomplete_batch(self):
        expected = {"a": "hash-a", "b": "hash-b"}
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            validate_results([result("a", "wrong"), result("b", "hash-b")], expected)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_results([result("a", "hash-a"), result("a", "hash-a")], expected)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_results([result("a", "hash-a")], expected)

    def test_legacy_coverage_survives_and_changed_version_stays_pending(self):
        legacy = self.root / "legacy"; legacy.mkdir()
        old_sha = "old-dataset-sha"
        inp = [{"Building_ID_建筑ID": "a", "_review": {"dataset_sha256": old_sha,
               "record_binding_sha256": binding(old_sha, "a")}}]
        (legacy / "old.jsonl").write_text("\n".join(json.dumps(x) for x in inp), encoding="utf-8")
        (legacy / "old_results.json").write_text(json.dumps([result("a", binding(old_sha, "a"))]), encoding="utf-8")
        self.assertEqual(import_legacy(self.state, legacy), 1)
        self.assertEqual(import_legacy(self.state, legacy), 0)
        imported = status(self.state)
        self.assertEqual(imported["unique_record_ids_reviewed_across_snapshots"], 1)
        self.assertEqual(imported["review_modes"], {"legacy_not_recorded": 1})
        # a changed record set/version has new bindings for every row.
        self.write_dataset(["a", "b", "c"], [9, 1, 2])
        view = status(self.state, self.dataset)["current_dataset"]
        self.assertEqual((view["reviewed"], view["pending"]), (0, 3))
        report = status(self.state, self.dataset)
        self.assertEqual(report["dataset_inventory"], {"records": 3, "unique_ids_never_reviewed_across_snapshots": 2})
        self.assertEqual(report["current_snapshot_pending"], 3)
        # Across versions, a known ID is excluded by default; an explicit ID
        # request can revalidate it against the new record binding.
        default_batch = export_batch(self.dataset, self.state, "next", 3)
        default_ids = [json.loads(x)["Building_ID_建筑ID"] for x in default_batch.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(default_ids, ["a", "b", "c"])

    def _complete(self, batch_id):
        path = self.state / f"{batch_id}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        results = [result(row["Building_ID_建筑ID"], row["_review"]["record_binding_sha256"], "not_retrieved")
                   for row in rows]
        result_path = self.root / f"{batch_id}_results.json"
        result_path.write_text(json.dumps(results), encoding="utf-8")
        ingest(self.state, batch_id, result_path)

    def test_unchanged_content_carries_review_and_new_attribute_invalidates(self):
        export_batch(self.dataset, self.state, "original", 2)
        self._complete("original")
        self.write_dataset(["a", "b"], [0, 99])
        progress = status(self.state, self.dataset)
        self.assertEqual(progress["current_dataset"]["reviewed_exact_snapshot"], 0)
        self.assertEqual(progress["current_dataset"]["carried_unchanged_rows"], 1)
        self.assertEqual(progress["current_snapshot_pending"], 1)
        self.assertEqual(progress["unique_record_ids_reviewed_across_snapshots"], 2)
        batch = export_batch(self.dataset, self.state, "changed", 2)
        selected = [json.loads(line)["Building_ID_建筑ID"] for line in batch.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(selected, ["b"])
        self.write_dataset(["a", "b"], [0, 99])
        frame = pd.read_parquet(self.dataset); frame["new_public_field"] = ["x", "y"]; frame.to_parquet(self.dataset)
        after_column = status(self.state, self.dataset)
        self.assertEqual(after_column["current_dataset"]["carried_unchanged_rows"], 0)
        self.assertEqual(after_column["current_snapshot_pending"], 2)

    def test_geometry_change_invalidates_only_changed_record(self):
        import geopandas as gpd
        from shapely.geometry import box

        self.dataset = self.root / "tiny_geo.parquet"
        frame = gpd.GeoDataFrame({"Building_ID_建筑ID": ["a", "b"], "name": ["一", "二"]},
                                 geometry=[box(0, 0, 10, 10), box(20, 20, 30, 30)], crs="EPSG:2326")
        frame.to_parquet(self.dataset)
        export_batch(self.dataset, self.state, "geo_original", 2)
        self._complete("geo_original")
        frame.loc[1, "geometry"] = box(21, 20, 31, 30)
        frame.to_parquet(self.dataset)
        progress = status(self.state, self.dataset)
        self.assertEqual(progress["current_dataset"]["carried_unchanged_rows"], 1)
        self.assertEqual(progress["current_snapshot_pending"], 1)

    def test_source_retrieval_status_must_be_honest(self):
        with self.assertRaisesRegex(ValueError, "retrieval status"):
            validate_results([result("a", "hash-a", "looked_at")], {"a": "hash-a"})

    def test_no_web_mode_requires_reason_and_cannot_claim_opened_sources(self):
        no_web = result("a", "hash-a", "historical_reference")
        no_web.update({"review_mode": "attribute_consistency_only_no_web", "queries_used": [],
                       "query_not_run_reason": "No reliable name or address supports a useful external search."})
        self.assertEqual(validate_results([no_web], {"a": "hash-a"}), [no_web])
        missing_reason = dict(no_web); missing_reason.pop("query_not_run_reason")
        with self.assertRaisesRegex(ValueError, "query_not_run_reason"):
            validate_results([missing_reason], {"a": "hash-a"})
        opened = dict(no_web); opened["sources"] = [{"url": "https://example.org", "retrieval_status": "opened_this_pass"}]
        with self.assertRaisesRegex(ValueError, "cannot claim an opened source"):
            validate_results([opened], {"a": "hash-a"})
        no_query_research = result("a", "hash-a")
        no_query_research["queries_used"] = []
        with self.assertRaisesRegex(ValueError, "source research requires actual queries"):
            validate_results([no_query_research], {"a": "hash-a"})
        verified = result("a", "hash-a")
        verified["fully_fact_verified"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim full fact verification"):
            validate_results([verified], {"a": "hash-a"})

    def test_pending_batches_reserve_ids_and_inputs_are_immutable(self):
        first = export_batch(self.dataset, self.state, "first", 2)
        first_ids = {json.loads(x)["Building_ID_建筑ID"] for x in first.read_text(encoding="utf-8").splitlines()}
        second = export_batch(self.dataset, self.state, "second", 2)
        second_ids = {json.loads(x)["Building_ID_建筑ID"] for x in second.read_text(encoding="utf-8").splitlines()}
        self.assertFalse(first_ids & second_ids)
        with first.open("a", encoding="utf-8") as f:
            f.write("{}\n")
        results_path = self.root / "results.json"
        results_path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input.*changed"):
            ingest(self.state, "first", results_path)

    def test_invalid_limit_and_unknown_requested_id_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            export_batch(self.dataset, self.state, "zero", 0)
        ids_file = self.root / "unknown.csv"; ids_file.write_text("missing\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "absent"):
            export_batch(self.dataset, self.state, "unknown", 1, ids_file)

    def test_compact_packet_tampering_rejects_ingest(self):
        export_batch(self.dataset, self.state, "packet", 1)
        compact = self.state / "packet.compact.json"
        compact.write_text(compact.read_text(encoding="utf-8") + " ", encoding="utf-8")
        results = self.root / "results.json"; results.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "compact inspection packet.*changed"):
            ingest(self.state, "packet", results)


if __name__ == "__main__":
    unittest.main()
