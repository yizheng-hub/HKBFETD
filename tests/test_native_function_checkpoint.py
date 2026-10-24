import copy
import hashlib
import json
import tempfile
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_function_checkpoint import NATIVE_BACKEND, NATIVE_MODEL, canonical_sha256, load_checkpoint

UNKNOWN = "Unknown_未知类别"
COMMERCIAL = "Commercial_商业类别"
OFFICE = "Office_办公室"
TAXONOMY = {OFFICE: COMMERCIAL}
MAIN = {UNKNOWN, COMMERCIAL}


def canonical_main(value):
    aliases = {"Unknown": UNKNOWN, "Commercial": COMMERCIAL, COMMERCIAL: COMMERCIAL, UNKNOWN: UNKNOWN}
    return aliases.get(value, UNKNOWN)


def canonical_sub(value):
    aliases = {"Unknown": UNKNOWN, "Office": OFFICE, OFFICE: OFFICE, UNKNOWN: UNKNOWN}
    return aliases.get(value, UNKNOWN)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_case(main=COMMERCIAL, sub=OFFICE, confidence=.91, approved=True, bid="42", batch_id="classifier_test"):
    public = {"BUILDINGSTRUCTUREID": bid, "BUILDINGNAMEEN": "Test Hall", "GFA": 123.4}
    for index in range(187):
        public[f"source_field_{index:03d}"] = None
    geometry = {"wkb_sha256": "a" * 64, "crs": "EPSG:2326", "component_scope": {"BUILDINGSTRUCTUREID": bid, "CSDI_Component_Use_Type": ""}}
    dataset = "d" * 64
    binding = sha((dataset + ":" + bid).encode())
    prompt_text = "instruction\n" + json.dumps({"evidence": public}, ensure_ascii=False)
    prompt_version = "test-prompt-v1"
    input_digest = canonical_sha256({"prompt_version": prompt_version, "model": NATIVE_MODEL, "prompt": prompt_text})
    request = {"building_id": bid, "record_binding_sha256": binding, "Input_SHA256": input_digest,
               "model": NATIVE_MODEL, "prompt_version": prompt_version, "prompt": prompt_text}
    response = {"building_id": bid, "record_binding_sha256": binding, "model": NATIVE_MODEL,
                "fact_verified": False,
                "review_context": {"context_sha256": "b"*64, "policy_sha256": sha(b"keyword policy\n"), "policy_note": "captured root policy"},
                "classifier_response": {"Input_SHA256": input_digest, "model": NATIVE_MODEL,
                                        "LLM_Main_Class": main, "LLM_Sub_Class": sub,
                                        "LLM_Confidence": confidence, "LLM_Reasoning": "test"}}
    reqbytes = json.dumps([request], ensure_ascii=False, separators=(",", ":"))
    originalbytes = reqbytes
    compactbytes = json.dumps([{"building_id": bid, "record_binding_sha256": binding, "Input_SHA256": input_digest,
                               "model": NATIVE_MODEL, "evidence": public}], ensure_ascii=False, separators=(",", ":"))
    jsonlbytes = json.dumps({"BUILDING_ID": bid, "_review": {"record_binding_sha256": binding, "dataset_sha256": dataset,
                             "batch_id": batch_id}}, ensure_ascii=False) + "\n"
    compactpacketbytes = json.dumps({"ids": [bid], "public_columns": list(public), "public_rows": [list(public.values())]}, ensure_ascii=False)
    resbytes = json.dumps([response], ensure_ascii=False, separators=(",", ":"))
    snapshot = {"dataset_sha256": dataset, "source_csv_sha256": "e" * 64, "stage3_geojson_sha256": "f" * 64, "historical_public_parquet_sha256": "9" * 64}
    keyword_text, clarification_text = "keyword policy\n", "root clarification\n"
    policy = {"keywords_utf8": keyword_text, "keywords_sha256": sha(keyword_text.encode()),
              "clarification_utf8": clarification_text, "clarification_sha256": sha(clarification_text.encode()),
              "taxonomy": TAXONOMY, "taxonomy_sha256": canonical_sha256(TAXONOMY)}
    guard = {"public_record": public, "source_row_sha256": canonical_sha256(public), "geometry": geometry,
             "record_binding_sha256": binding, "source_snapshot_sha256": snapshot["source_csv_sha256"]}
    acceptance = {"batch": batch_id, "root_actual_semantic_read_all32_responses_and_request_evidence": True,
                  "result_sha256": sha(resbytes.encode()), "basis": dataset, "native_requested_model": NATIVE_MODEL,
                  "native_backend": "Interactive native session, not an external API response"}
    accbytes = json.dumps(acceptance, ensure_ascii=False, separators=(",", ":"))
    batch = {"batch_id": batch_id, "backend": NATIVE_BACKEND, "dataset_sha256": dataset,
             "review_context": {"policy": "captured test context", "prompt_count": 1},
             "requests_utf8": reqbytes, "request_sha256": sha(reqbytes.encode()),
             "original_requests_utf8": originalbytes, "original_requests_sha256": sha(originalbytes.encode()),
             "compact_requests_utf8": compactbytes, "compact_requests_sha256": sha(compactbytes.encode()),
             "compact_packet_utf8": compactpacketbytes, "compact_packet_sha256": sha(compactpacketbytes.encode()),
             "jsonl_utf8": jsonlbytes, "jsonl_sha256": sha(jsonlbytes.encode()), "results_utf8": resbytes, "result_sha256": sha(resbytes.encode()),
             "root_acceptance_utf8": accbytes, "root_acceptance_sha256": sha(accbytes.encode())}
    batch_packet_sha = batch["compact_packet_sha256"]
    jsonl_sha = batch["jsonl_sha256"]
    guard["context_sha256"] = canonical_sha256({"request": request, "actual_compact_public_packet_sha256": batch_packet_sha,
        "original_jsonl_sha256": jsonl_sha, "original_requests_sha256": batch["original_requests_sha256"],
        "compact_requests_sha256": batch["compact_requests_sha256"], "review_context": response["review_context"], "policy": policy,
        "source_snapshot": snapshot, "record_binding_sha256": request["record_binding_sha256"], "geometry": geometry})
    artifact = {"schema_version": 1, "backend": NATIVE_BACKEND, "model": NATIVE_MODEL, "policy": policy,
                "source_snapshot": snapshot, "batches": [batch], "row_guards": {bid: guard},
                "row_approvals": {}}
    if approved:
        approval = {"decision": "approved", "building_id": bid, "batch_id": batch["batch_id"],
                    "context_sha256": guard["context_sha256"], "root_acceptance_sha256": batch["root_acceptance_sha256"],
                    "result_sha256": batch["result_sha256"], "record_binding_sha256": binding,
                    "model": NATIVE_MODEL, "backend": NATIVE_BACKEND}
        approval_raw = json.dumps(approval, ensure_ascii=False, separators=(",", ":"))
        artifact["row_approvals"][bid] = {"approval_utf8": approval_raw, "approval_sha256": sha(approval_raw.encode())}
    return artifact, public, geometry


class NativeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "checkpoint.json"
        self.artifact, self.public, self.geometry = make_case()
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.artifact, ensure_ascii=False), encoding="utf-8")

    def test_step4_categorical_code_serialization_matches_string_source_packets(self):
        import importlib.util
        step4_path = Path(__file__).resolve().parents[1] / "src" / "4.llm_verification.py"
        spec = importlib.util.spec_from_file_location("candidate_step4_prompt_test", step4_path)
        step4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(step4)
        prompt = step4.generate_prompt({"CATEGORY": 1})
        evidence = json.loads(prompt.split("\n", 1)[1])["evidence"]
        self.assertEqual(evidence["CATEGORY"], "1")

    def test_step4_root_rejected_response_is_finished_without_adoption(self):
        import importlib.util
        import pandas as pd
        from unittest.mock import patch
        step4_path = Path(__file__).resolve().parents[1] / "src" / "4.llm_verification.py"
        spec = importlib.util.spec_from_file_location("step4_root_rejected_queue_test", step4_path)
        step4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(step4)
        base_path = Path(self.tmp.name) / "rules.csv"
        cache_path = Path(self.tmp.name) / "cache.csv"
        pd.DataFrame([{"BUILDINGSTRUCTUREID": "145291", "BUILDINGNAMEEN": "Cathay Dragon House",
                       "BUILDINGNAMETC": "", "Matched_POIs": "", "Final_Main_Class": COMMERCIAL,
                       "Final_Sub_Class": UNKNOWN}]).to_csv(base_path, index=False)
        terminal = "reviewed_not_adopted_root_rejected"
        with patch.multiple(step4, RULE_ENGINE_OUTPUT_PATH=str(base_path),
                            LLM_VERIFICATION_OUTPUT=str(cache_path), LEGACY_OUTPUT_DIR=self.tmp.name), \
                patch.object(step4, "native_results_for_base", return_value={"145291": {"status": terminal, "adoptable": False}}):
            step4.main()
        current = pd.read_csv(cache_path.with_name("step4_current_status.csv"))
        pending = pd.read_csv(cache_path.with_name("step4_pending_review.csv"))
        self.assertEqual(current.loc[0, "LLM_Status"], terminal)
        self.assertTrue(pending.empty)
        self.assertTrue(pd.read_csv(cache_path).empty)

    def load(self, artifact_path=None, row=None, geom=None, keywords=sha(b"keyword policy\n"), clarification=sha(b"root clarification\n"),
             canonical_sub_fn=canonical_sub, prompt_evidence_fn=None):
        return load_checkpoint(artifact_path or self.path, {"42": row or self.public}, {"42": geom or self.geometry},
            "e"*64, "f"*64, keywords, clarification, canonical_main, canonical_sub_fn, TAXONOMY,
            prompt_evidence=prompt_evidence_fn or (lambda current: self.public))


    def test_batch_basis_is_independent_of_global_historical_parquet_basis(self):
        self.artifact, self.public, self.geometry = make_case()
        self.artifact["source_snapshot"]["dataset_sha256"] = "9" * 64
        request = json.loads(self.artifact["batches"][0]["requests_utf8"])[0]
        result = json.loads(self.artifact["batches"][0]["results_utf8"])[0]
        batch = self.artifact["batches"][0]
        guard = self.artifact["row_guards"]["42"]
        guard["context_sha256"] = canonical_sha256({"request": request, "actual_compact_public_packet_sha256": batch["compact_packet_sha256"],
            "original_jsonl_sha256": batch["jsonl_sha256"], "original_requests_sha256": batch["original_requests_sha256"],
            "compact_requests_sha256": batch["compact_requests_sha256"],
            "review_context": result["review_context"], "policy": self.artifact["policy"],
            "source_snapshot": self.artifact["source_snapshot"], "record_binding_sha256": request["record_binding_sha256"],
            "geometry": guard["geometry"]})
        approval = {"decision": "approved", "building_id": "42", "batch_id": batch["batch_id"],
                    "context_sha256": guard["context_sha256"], "root_acceptance_sha256": batch["root_acceptance_sha256"],
                    "result_sha256": batch["result_sha256"], "record_binding_sha256": request["record_binding_sha256"],
                    "model": NATIVE_MODEL, "backend": NATIVE_BACKEND}
        approval_raw = json.dumps(approval, ensure_ascii=False, separators=(",", ":"))
        self.artifact["row_approvals"]["42"] = {"approval_utf8": approval_raw, "approval_sha256": sha(approval_raw.encode())}
        self.write()
        self.assertEqual(self.load()["42"]["status"], "matching_native_checkpoint")

    def test_two_synthetic_batches_preserve_each_original_prompt_binding(self):
        first, public1, geometry1 = make_case(COMMERCIAL, OFFICE, .91, False, "42", "classifier_test_a")
        second, public2, geometry2 = make_case(UNKNOWN, UNKNOWN, .31, False, "43", "classifier_test_b")
        artifact = first
        artifact["batches"].append(second["batches"][0])
        artifact["row_guards"].update(second["row_guards"])
        artifact["row_approvals"].update(second["row_approvals"])
        self.write()
        self.path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
        prompts = {
            "42": json.loads(first["batches"][0]["requests_utf8"])[0]["prompt"],
            "43": json.loads(second["batches"][0]["requests_utf8"])[0]["prompt"],
        }
        source_snapshot = artifact["source_snapshot"]
        result = load_checkpoint(self.path, {"42": public1, "43": public2},
            {"42": geometry1, "43": geometry2}, source_snapshot["source_csv_sha256"],
            source_snapshot["stage3_geojson_sha256"], artifact["policy"]["keywords_sha256"],
            artifact["policy"]["clarification_sha256"], canonical_main, canonical_sub, TAXONOMY,
            prompt_builder=lambda row: prompts[str(row["BUILDINGSTRUCTUREID"])], prompt_version="test-prompt-v1")
        self.assertEqual(result["42"]["status"], "pending_root_adoption")
        self.assertEqual(result["43"]["status"], "reviewed_unknown")

    def test_explicit_root_approval_adopts_complete_canonical_response(self):
        result = self.load()["42"]
        self.assertEqual(result["status"], "matching_native_checkpoint")
        self.assertTrue(result["adoptable"])

    def test_unknown_partial_and_low_confidence_remain_reviewed_but_not_adopted(self):
        for main, sub, conf, expected in [(UNKNOWN, UNKNOWN, .95, "reviewed_unknown"),
                                           (COMMERCIAL, UNKNOWN, .95, "reviewed_partial"),
                                           (COMMERCIAL, OFFICE, .79, "reviewed_low_confidence")]:
            self.artifact, self.public, self.geometry = make_case(main, sub, conf)
            self.write()
            result = self.load()["42"]
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["adoptable"])

    def test_original_subclass_alias_uses_explicit_canonicalizer(self):
        self.artifact, self.public, self.geometry = make_case(COMMERCIAL, "Office", .9)
        self.write()
        result = self.load()["42"]
        self.assertEqual(result["sub"], OFFICE)
        self.assertEqual(result["status"], "matching_native_checkpoint")

    def test_missing_root_row_approval_stays_pending(self):
        self.artifact, self.public, self.geometry = make_case(approved=False)
        self.write()
        self.assertEqual(self.load()["42"]["status"], "pending_root_adoption")

    def test_source_geometry_policy_model_backend_acceptance_tampering_rejected(self):
        for mutate in [
            lambda a: a["source_snapshot"].update(source_csv_sha256="0"*64),
            lambda a: a["source_snapshot"].update(stage3_geojson_sha256="0"*64),
            lambda a: a["policy"].update(keywords_sha256="0"*64),
            lambda a: a["policy"].update(taxonomy_sha256="0"*64),
            lambda a: a.update(model="different-model"),
            lambda a: a["batches"][0].update(backend="api"),
            lambda a: a["batches"][0].update(root_acceptance_utf8="{}"),
        ]:
            self.artifact, self.public, self.geometry = make_case()
            mutate(self.artifact); self.write()
            with self.assertRaises(ValueError): self.load()

    def test_compact_public_packet_bytes_are_hash_bound(self):
        self.artifact["batches"][0]["compact_packet_utf8"] += " "
        self.write()
        with self.assertRaisesRegex(ValueError, "compact_packet_utf8 hash mismatch"):
            self.load()

    def test_invalid_per_record_approval_never_promotes(self):
        approval = json.loads(self.artifact["row_approvals"]["42"]["approval_utf8"])
        approval["result_sha256"] = "0" * 64
        raw = json.dumps(approval, ensure_ascii=False, separators=(",", ":"))
        self.artifact["row_approvals"]["42"] = {"approval_utf8": raw, "approval_sha256": sha(raw.encode())}
        self.write()
        self.assertEqual(self.load()["42"]["status"], "invalid_root_approval")

    def test_stale_row_geometry_and_prompt_evidence_do_not_adopt(self):
        changed = dict(self.public, GFA=124)
        self.assertEqual(self.load(row=changed)["42"]["status"], "stale_native_context")
        changed = dict(self.public, unbound_source_field="unexpected")
        self.assertEqual(self.load(row=changed)["42"]["status"], "stale_native_context")
        changed_geom = dict(self.geometry, wkb_sha256="9"*64)
        self.assertEqual(self.load(geom=changed_geom)["42"]["status"], "stale_native_context")
        self.assertEqual(self.load(prompt_evidence_fn=lambda current: {"wrong": True})["42"]["status"], "stale_native_context")
        with self.assertRaises(ValueError): self.load(clarification="0"*64)


    def test_root_rejection_is_a_terminal_nonadoption_status_and_keeps_response(self):
        self.artifact, self.public, self.geometry = make_case(COMMERCIAL, OFFICE, .87, True, "145291", "classifier_00008")
        guard = self.artifact["row_guards"]["145291"]
        batch = self.artifact["batches"][0]
        approval = {"decision": "rejected", "building_id": "145291", "batch_id": batch["batch_id"],
                    "context_sha256": guard["context_sha256"], "root_acceptance_sha256": batch["root_acceptance_sha256"],
                    "result_sha256": batch["result_sha256"], "record_binding_sha256": guard["record_binding_sha256"],
                    "model": NATIVE_MODEL, "backend": NATIVE_BACKEND}
        raw = json.dumps(approval, ensure_ascii=False, separators=(",", ":"))
        self.artifact["row_approvals"]["145291"] = {"approval_utf8": raw, "approval_sha256": sha(raw.encode())}
        self.write()
        result = load_checkpoint(self.path, {"145291": self.public}, {"145291": self.geometry},
            "e"*64, "f"*64, sha(b"keyword policy\n"), sha(b"root clarification\n"),
            canonical_main, canonical_sub, TAXONOMY)["145291"]
        self.assertEqual(result["status"], "reviewed_not_adopted_root_rejected")
        self.assertFalse(result["adoptable"])
        self.assertEqual(result["response"]["classifier_response"]["LLM_Sub_Class"], OFFICE)

    def test_root_rejection_with_wrong_context_binding_is_invalid(self):
        self.artifact, self.public, self.geometry = make_case(COMMERCIAL, OFFICE, .87, True, "145291", "classifier_00008")
        batch = self.artifact["batches"][0]
        approval = {"decision": "rejected", "building_id": "145291", "batch_id": batch["batch_id"],
                    "context_sha256": "0"*64, "root_acceptance_sha256": batch["root_acceptance_sha256"],
                    "result_sha256": batch["result_sha256"], "record_binding_sha256": self.artifact["row_guards"]["145291"]["record_binding_sha256"],
                    "model": NATIVE_MODEL, "backend": NATIVE_BACKEND}
        raw = json.dumps(approval, ensure_ascii=False, separators=(",", ":"))
        self.artifact["row_approvals"]["145291"] = {"approval_utf8": raw, "approval_sha256": sha(raw.encode())}
        self.write()
        result = load_checkpoint(self.path, {"145291": self.public}, {"145291": self.geometry},
            "e"*64, "f"*64, sha(b"keyword policy\n"), sha(b"root clarification\n"),
            canonical_main, canonical_sub, TAXONOMY)["145291"]
        self.assertEqual(result["status"], "invalid_root_approval")

    def test_replacement_character_label_is_stale_not_promotable(self):
        self.artifact["batches"][0]["results_utf8"] = self.artifact["batches"][0]["results_utf8"].replace("Office", "Office\ufffd")
        self.artifact["batches"][0]["result_sha256"] = sha(self.artifact["batches"][0]["results_utf8"].encode())
        result_rows = json.loads(self.artifact["batches"][0]["results_utf8"])
        acceptance = json.loads(self.artifact["batches"][0]["root_acceptance_utf8"])
        acceptance["result_sha256"] = self.artifact["batches"][0]["result_sha256"]
        raw_acceptance = json.dumps(acceptance, ensure_ascii=False, separators=(",", ":"))
        self.artifact["batches"][0]["root_acceptance_utf8"] = raw_acceptance
        self.artifact["batches"][0]["root_acceptance_sha256"] = sha(raw_acceptance.encode())
        self.artifact["row_approvals"] = {}
        self.write()
        self.assertEqual(self.load()["42"]["status"], "stale_native_context")


    def test_prompt_instruction_and_version_changes_with_same_evidence_are_stale(self):
        for field, value in (("prompt", "Changed instruction.\n" + json.dumps({"evidence": self.public}, ensure_ascii=False)),
                             ("prompt_version", "different-prompt-version")):
            self.artifact, self.public, self.geometry = make_case()
            batch = self.artifact["batches"][0]
            requests = json.loads(batch["requests_utf8"])
            requests[0][field] = value
            raw = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
            batch["requests_utf8"] = raw
            batch["request_sha256"] = sha(raw.encode())
            self.write()
            self.assertEqual(self.load()["42"]["status"], "stale_native_context")

    def test_invalid_taxonomy_and_bad_confidence_never_adopt(self):
        self.artifact, self.public, self.geometry = make_case(COMMERCIAL, "Unmapped", .95)
        self.write()
        self.assertEqual(self.load()["42"]["status"], "reviewed_invalid_taxonomy")
        self.artifact, self.public, self.geometry = make_case(COMMERCIAL, OFFICE, float("nan"))
        self.write()
        self.assertEqual(self.load()["42"]["status"], "reviewed_low_confidence")
        self.artifact, self.public, self.geometry = make_case(COMMERCIAL, OFFICE, 1.01)
        self.write()
        self.assertEqual(self.load()["42"]["status"], "reviewed_low_confidence")


if __name__ == "__main__":
    unittest.main()
