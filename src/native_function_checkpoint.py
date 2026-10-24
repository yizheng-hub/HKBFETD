from __future__ import annotations
import hashlib, json, math
from pathlib import Path

NATIVE_BACKEND = "interactive_native_session"
NATIVE_MODEL = "gpt-6-luna"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_rows(raw: str, key: str):
    obj=json.loads(raw)
    rows=obj.get(key) if isinstance(obj,dict) else obj
    if not isinstance(rows,list): raise ValueError(f"invalid {key} packet")
    return rows


def resolve_occurrences(batches, supersessions):
    """Return ID -> (batch, request, result). Keep all packets intact; allow one declared replacement only."""
    by_name={}
    occurrences={}
    for batch in batches:
        name=batch.get("batch_id")
        if not isinstance(name,str) or name in by_name: raise ValueError("duplicate or missing batch_id")
        by_name[name]=batch
        for field,key,hash_key in (("requests_utf8","requests","request_sha256"),("results_utf8","results","result_sha256")):
            raw=batch.get(field)
            if not isinstance(raw,str) or _sha(raw)!=batch.get(hash_key):
                raise ValueError(f"{name} {field} SHA mismatch")
        request_rows=_raw_rows(batch["requests_utf8"],"requests")
        result_rows=_raw_rows(batch["results_utf8"],"results")
        req={str(r["building_id"]):r for r in request_rows}
        res={str(r["building_id"]):r for r in result_rows}
        if len(req)!=len(request_rows) or len(res)!=len(result_rows) or set(req)!=set(res):
            raise ValueError(f"{name} request/result IDs differ or duplicate")
        for bid,request in req.items():
            response=res[bid]
            if request.get("Input_SHA256") != response.get("classifier_response",{}).get("Input_SHA256"):
                raise ValueError(f"{name}/{bid} response does not bind request Input_SHA256")
            occurrences.setdefault(bid,[]).append((batch,request,response))
    declared={}
    for entry in supersessions:
        bid=str(entry.get("building_id"))
        if bid in declared: raise ValueError("multiple supersessions for one ID")
        declared[bid]=entry
    out={}
    for bid,items in occurrences.items():
        entry=declared.get(bid)
        if len(items)==1:
            if entry is not None: raise ValueError("supersession declared for non-duplicate ID")
            out[bid]=items[0]; continue
        if len(items)!=2 or entry is None: raise ValueError("duplicate response must have exactly one explicit supersession")
        first,second=items
        prior_id=entry.get("prior_batch_id"); current_id=entry.get("current_batch_id")
        if first[0].get("batch_id")!=prior_id or second[0].get("batch_id")!=current_id:
            raise ValueError("supersession must follow original batch order exactly")
        if first[0].get("result_sha256")!=entry.get("prior_result_sha256") or second[0].get("result_sha256")!=entry.get("current_result_sha256"):
            raise ValueError("supersession result SHA mismatch")
        if first[1].get("Input_SHA256")!=entry.get("prior_input_sha256") or second[1].get("Input_SHA256")!=entry.get("current_input_sha256"):
            raise ValueError("supersession Input_SHA256 mismatch")
        if first[1].get("Input_SHA256")==second[1].get("Input_SHA256"):
            raise ValueError("same-input replacement is forbidden")
        if list(by_name).index(prior_id)>=list(by_name).index(current_id):
            raise ValueError("reversed supersession order")
        out[bid]=second
        declared.pop(bid)
    if declared: raise ValueError("supersession has no matching duplicate occurrence")
    return out





def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _read_raw_json(text, label):
    if not isinstance(text, str):
        raise ValueError(f"{label} must be embedded as original UTF-8 text")
    try:
        return json.loads(text)
    except Exception as exc:
        raise ValueError(f"invalid embedded {label}") from exc


def _rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("requests"), list):
        return value["requests"]
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return value["results"]
    raise ValueError("embedded batch request/result must contain row lists")


def _unique_map(parsed, label):
    rows = _rows(parsed)
    result = {str(r["building_id"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate building_id in embedded {label}")
    return result


def _request_map(parsed):
    return _unique_map(parsed, "requests")


def _response_map(parsed):
    return _unique_map(parsed, "results")


def captured_review_context(response, batch, policy_sha256):
    """Bind the exact context shape recorded with a response without overstating its provenance."""
    captured = response.get("review_context")
    if isinstance(captured, dict) and captured.get("policy_sha256") and captured.get("context_sha256"):
        return captured
    if captured is None:
        return batch.get("captured_root_review_context")
    return {"context_sha256": canonical_sha256(captured), "policy_sha256": policy_sha256,
            "context_source": "captured_result_review_context_metadata",
            "captured_context": captured}


def classify_status(response, adoption_decision, canonical_main, canonical_sub, taxonomy, confidence_floor=.80):
    raw_main = response.get("LLM_Main_Class")
    raw_sub = response.get("LLM_Sub_Class")
    main = canonical_main(raw_main)
    sub = canonical_sub(raw_sub)
    try:
        confidence = float(response.get("LLM_Confidence"))
    except (TypeError, ValueError):
        confidence = float("nan")
    if main == "Unknown_未知类别" and raw_main not in (None, "", "Unknown", "Unknown_未知类别", "未知类别"):
        return "reviewed_invalid_taxonomy", main, sub, confidence, False
    if main == "Unknown_未知类别":
        return "reviewed_unknown", main, sub, confidence, False
    if raw_sub not in ("Unknown_未知类别", "Unknown", None, "") and sub == "Unknown_未知类别":
        return "reviewed_invalid_taxonomy", main, sub, confidence, False
    if sub == "Unknown_未知类别":
        return "reviewed_partial", main, sub, confidence, False
    if not math.isfinite(confidence) or confidence < confidence_floor or confidence > 1 or confidence < 0:
        return "reviewed_low_confidence", main, sub, confidence, False
    if main != "Mixed-use_混合用途" and taxonomy.get(sub) != main:
        return "reviewed_invalid_taxonomy", main, sub, confidence, False
    if adoption_decision != "root_approved":
        return "pending_root_adoption", main, sub, confidence, False
    return "matching_native_checkpoint", main, sub, confidence, True


def load_checkpoint(path, current_rows, current_geometry, source_csv_sha256,
                    stage3_geojson_sha256, keywords_sha256, clarification_sha256,
                    canonical_main, canonical_sub, taxonomy, confidence_floor=.80,
                    prompt_evidence=None, prompt_builder=None, prompt_version=None):
    """Validate and replay a frozen native checkpoint without network/model calls.

    current_rows maps building_id to the complete source-stage public row (all guarded columns).
    current_geometry maps id to {wkb_sha256, crs, component_scope}.
    """
    path = Path(path)
    if not path.is_file():
        return {bid: {"status": "pending_no_matching_checkpoint"} for bid in current_rows}
    artifact_bytes = path.read_bytes()
    artifact = json.loads(artifact_bytes.decode("utf-8"))
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported native checkpoint schema")
    if artifact.get("backend") != NATIVE_BACKEND or artifact.get("model") != NATIVE_MODEL:
        raise ValueError("native backend/model mismatch")
    policy = artifact.get("policy", {})
    if (policy.get("keywords_sha256") != keywords_sha256
            or policy.get("clarification_sha256") != clarification_sha256
            or sha256_bytes(policy.get("keywords_utf8", "").encode("utf-8")) != keywords_sha256
            or sha256_bytes(policy.get("clarification_utf8", "").encode("utf-8")) != clarification_sha256
            or policy.get("taxonomy") != taxonomy
            or policy.get("taxonomy_sha256") != canonical_sha256(taxonomy)):
        raise ValueError("native policy/context mismatch")
    snapshot = artifact.get("source_snapshot", {})
    if (snapshot.get("source_csv_sha256") != source_csv_sha256
            or snapshot.get("stage3_geojson_sha256") != stage3_geojson_sha256):
        raise ValueError("native source snapshot stale")
    batches = {}
    batch_acceptances = {}
    request_original_prompt_match = {}
    for batch in artifact.get("batches", []):
        raw_parts = (("requests_utf8", "request_sha256"), ("results_utf8", "result_sha256"), ("root_acceptance_utf8", "root_acceptance_sha256"),
                     ("original_requests_utf8", "original_requests_sha256"), ("compact_requests_utf8", "compact_requests_sha256"),
                     ("compact_packet_utf8", "compact_packet_sha256"), ("jsonl_utf8", "jsonl_sha256"))
        for part, hash_key in raw_parts:
            raw = batch.get(part)
            if not isinstance(raw, str) or sha256_bytes(raw.encode("utf-8")) != batch.get(hash_key):
                raise ValueError(f"native embedded {part} hash mismatch")
        preserved_raw = batch.get("preserved_original_results_utf8")
        if preserved_raw is not None:
            preserved_sha = sha256_bytes(preserved_raw.encode("utf-8"))
            if preserved_sha != batch.get("preserved_original_results_sha256"):
                raise ValueError("native preserved original result SHA mismatch")
            acceptance_for_original = _read_raw_json(batch.get("root_acceptance_utf8"), "root acceptance")
            if (acceptance_for_original.get("preserved_original_sha256")
                    and acceptance_for_original.get("preserved_original_sha256") != preserved_sha):
                raise ValueError("native preserved original result differs from root acceptance SHA")
        elif batch.get("preserved_original_results_status") == "embedded_and_sha_verified":
            raise ValueError("native preserved original result bytes are missing")
        requests = _read_raw_json(batch["requests_utf8"], "requests")
        original_requests = _read_raw_json(batch["original_requests_utf8"], "original requests")
        compact_requests = _read_raw_json(batch["compact_requests_utf8"], "compact requests")
        compact_packet = _read_raw_json(batch["compact_packet_utf8"], "original compact public packet")
        jsonl_rows = [_read_raw_json(line, "original JSONL request") for line in batch["jsonl_utf8"].splitlines() if line.strip()]
        results = _read_raw_json(batch["results_utf8"], "results")
        result_contexts = _response_map(results)
        context_missing_ids = sorted(bid for bid, row in result_contexts.items() if row.get("review_context") is None)
        context_incomplete_ids = sorted(bid for bid, row in result_contexts.items()
            if row.get("review_context") is not None and not (
                isinstance(row.get("review_context"), dict)
                and row["review_context"].get("policy_sha256")
                and row["review_context"].get("context_sha256")))
        context_unstructured_ids = sorted(set(context_missing_ids) | set(context_incomplete_ids))
        if (batch.get("historical_model_context_missingIDs") is not None
                and batch["historical_model_context_missingIDs"] != context_missing_ids):
            raise ValueError("native recorded missing-context IDs differ from original results")
        if (batch.get("historical_model_context_incompleteIDs") is not None
                and batch["historical_model_context_incompleteIDs"] != context_incomplete_ids):
            raise ValueError("native recorded incomplete-context IDs differ from original results")
        if (batch.get("historical_model_context_unstructuredIDs") is not None
                and batch["historical_model_context_unstructuredIDs"] != context_unstructured_ids):
            raise ValueError("native recorded unstructured-context IDs differ from original results")
        batch_basis = batch.get("dataset_sha256")
        if not isinstance(batch_basis, str) or len(batch_basis) != 64:
            raise ValueError("native batch lacks its own dataset basis")
        original_rows = _request_map(original_requests)
        compact_rows = _request_map(compact_requests)
        jsonl_by_binding = {}
        for row in jsonl_rows:
            review = row.get("_review", {})
            binding = review.get("record_binding_sha256")
            if not binding or binding in jsonl_by_binding:
                raise ValueError("original JSONL has missing or duplicate record binding")
            jsonl_by_binding[binding] = row
        compact_packet_ids = set(str(value) for value in compact_packet.get("ids", []))
        if (compact_packet_ids != set(original_rows) or set(original_rows) != set(compact_rows)
                or set(original_rows) != set(_request_map(requests)) or len(jsonl_rows) != len(original_rows)):
            raise ValueError("original, compact, and bound request IDs differ")
        for bid, req in _request_map(requests).items():
            full = original_rows[bid]
            compact = compact_rows[bid]
            jsonl_row = jsonl_by_binding.get(req.get("record_binding_sha256"), {})
            jsonl_review = jsonl_row.get("_review", {})
            if (jsonl_review.get("dataset_sha256") != batch_basis
                    or jsonl_review.get("batch_id") != batch.get("batch_id")
                    or full.get("record_binding_sha256") != req.get("record_binding_sha256")
                    or full.get("Input_SHA256") != req.get("Input_SHA256")
                    or full.get("model") != req.get("model")
                    or compact.get("record_binding_sha256") != req.get("record_binding_sha256")
                    or compact.get("Input_SHA256") != req.get("Input_SHA256")
                    or compact.get("evidence") != json.loads(full.get("prompt", "{}" ).split("\n", 1)[1]).get("evidence")):
                raise ValueError("original prompt and compact request do not bind exactly")
            request_original_prompt_match[bid] = (full.get("prompt") == req.get("prompt")
                                                  and full.get("prompt_version") == req.get("prompt_version"))
        for bid, response in _response_map(results).items():
            context = captured_review_context(response, batch, policy.get("keywords_sha256"))
            if not isinstance(context, dict) or not context.get("policy_sha256") or not context.get("context_sha256"):
                raise ValueError(f"native result lacks captured review context/root policy for {bid}")
            if response.get("review_context") is None and context.get("context_source") != "captured_root_semantic_acceptance_context":
                raise ValueError(f"native result fallback context is not identified as root semantic acceptance for {bid}")
            if context.get("policy_sha256") != policy.get("keywords_sha256"):
                raise ValueError(f"native result policy context mismatch for {bid}")
        acceptance = _read_raw_json(batch["root_acceptance_utf8"], "root acceptance")
        root_context = batch.get("captured_root_review_context")
        if root_context is not None:
            expected_policy_note_sha = canonical_sha256({
                "root_scope_clarification": acceptance.get("root_scope_clarification"),
                "scope_followups": acceptance.get("scope_followups")})
            if (root_context.get("context_source") != "captured_root_semantic_acceptance_context"
                    or root_context.get("provenance_type") != "root_adoption_context_not_original_model_context"
                    or root_context.get("context_sha256") != batch.get("root_acceptance_sha256")
                    or root_context.get("policy_sha256") != policy.get("keywords_sha256")
                    or root_context.get("policy_note_sha256") != expected_policy_note_sha
                    or root_context.get("captured_at_utc") != acceptance.get("accepted_utc")):
                raise ValueError("captured root-adoption context does not bind immutable acceptance and policy")
        if not acceptance.get("root_actual_semantic_read_all32_responses_and_request_evidence"):
            raise ValueError("batch lacks recorded root semantic acceptance")
        if acceptance.get("result_sha256") != batch["result_sha256"] or acceptance.get("basis") != batch_basis:
            raise ValueError("root acceptance does not bind this result/source basis")
        if acceptance.get("native_requested_model") != NATIVE_MODEL:
            raise ValueError("root acceptance model mismatch")
        if "interactive native" not in str(acceptance.get("native_backend", "")).lower():
            raise ValueError("root acceptance does not identify an interactive native backend")
        accepted_policy_sha = acceptance.get("taxonomy_sha256", acceptance.get("root_taxonomy_sha256"))
        if accepted_policy_sha and accepted_policy_sha != policy.get("keywords_sha256"):
            raise ValueError("root acceptance taxonomy hash mismatch")
        if batch.get("batch_id") != acceptance.get("batch") or batch.get("backend") != NATIVE_BACKEND:
            raise ValueError("native batch identity/backend mismatch")
        bid_requests, bid_results = _request_map(requests), _response_map(results)
        batch_acceptances[batch.get("batch_id")] = acceptance
        if set(bid_results) != set(bid_requests):
            raise ValueError("native requests and results must have the same exact ID set")
        for bid, response in bid_results.items():
            batches[bid] = (batch, acceptance, bid_requests[bid], response)
    # Duplicate IDs are resolved only through an exact, ordered, SHA-bound different-prompt declaration.
    resolved = resolve_occurrences(artifact.get("batches", []), artifact.get("supersessions", []))
    batches = {bid: (batch, batch_acceptances[batch["batch_id"]], request, response)
               for bid, (batch, request, response) in resolved.items()}
    output = {}
    for bid, current in current_rows.items():
        record = batches.get(str(bid))
        if not record:
            output[str(bid)] = {"status": "pending_no_matching_checkpoint"}
            continue
        batch, acceptance, request, result = record
        guard = artifact.get("row_guards", {}).get(str(bid), {})
        row_approval = artifact.get("row_approvals", {}).get(str(bid))
        adoption_decision = "not_root_approved"
        current_geom = current_geometry.get(str(bid), {})
        public_record = guard.get("public_record", {})
        current_public_record = {key: current.get(key) for key in public_record}
        batch_for_context = record[0]
        compact_packet_sha256 = batch_for_context.get("compact_packet_sha256")
        jsonl_sha256 = batch_for_context.get("jsonl_sha256")
        original_requests_sha256 = batch_for_context.get("original_requests_sha256")
        compact_requests_sha256 = batch_for_context.get("compact_requests_sha256")
        context_payload = {"request": request, "actual_compact_public_packet_sha256": compact_packet_sha256,
                           "original_jsonl_sha256": jsonl_sha256,
                           "original_requests_sha256": original_requests_sha256,
                           "compact_requests_sha256": compact_requests_sha256,
                           "review_context": captured_review_context(result, batch, policy.get("keywords_sha256")),
                           "policy": policy, "source_snapshot": snapshot,
                           "record_binding_sha256": request.get("record_binding_sha256") if request else None,
                           "geometry": guard.get("geometry")}
        context_matches = guard.get("context_sha256") == canonical_sha256(context_payload)
        prompt_matches = True
        if request is not None:
            prompt = request.get("prompt", "")
            try:
                supplied = json.loads(prompt.split("\n", 1)[1])["evidence"]
                if prompt_evidence is not None:
                    prompt_matches = supplied == prompt_evidence(current)
                if prompt_builder is not None:
                    prompt_matches = prompt_matches and prompt == prompt_builder(current)
                prompt_matches = prompt_matches and request_original_prompt_match.get(str(bid), False)
                if prompt_version is not None:
                    prompt_matches = prompt_matches and request.get("prompt_version") == prompt_version
                prompt_matches = prompt_matches and request.get("Input_SHA256") == canonical_sha256({
                    "prompt_version": request.get("prompt_version"), "model": NATIVE_MODEL, "prompt": prompt})
            except (IndexError, KeyError, ValueError, TypeError):
                prompt_matches = False
        expected_binding = sha256_bytes((str(batch.get("dataset_sha256")) + ":" + str(bid)).encode("utf-8"))
        guard_geometry = guard.get("geometry", {})
        stale_checks = {
            "missing_request": request is None,
            "request_model": request is not None and request.get("model") != NATIVE_MODEL,
            "source_guard_field_count": len(public_record) != 190,
            "geometry_incomplete": (not guard_geometry.get("wkb_sha256") or not guard_geometry.get("crs") or not guard_geometry.get("component_scope")),
            "record_binding": request is not None and request.get("record_binding_sha256") != expected_binding,
            "input_hash_missing": request is None or len(str(request.get("Input_SHA256", ""))) != 64,
            "response_model": result.get("classifier_response", {}).get("model") != NATIVE_MODEL,
            "replacement_character_label": "\ufffd" in str(result.get("classifier_response", {}).get("LLM_Main_Class", "")) or "\ufffd" in str(result.get("classifier_response", {}).get("LLM_Sub_Class", "")),
            "source_row_hash": guard.get("source_row_sha256") != canonical_sha256(public_record),
            "source_row_changed": set(public_record) != set(current) or public_record != current_public_record,
            "geometry_changed": guard.get("geometry") != current_geom,
            "context_changed": not context_matches,
            "prompt_changed": not prompt_matches,
            "guard_binding_changed": request is not None and request.get("record_binding_sha256") != guard.get("record_binding_sha256"),
            "response_binding_changed": result.get("record_binding_sha256") != guard.get("record_binding_sha256"),
            "source_csv_changed": guard.get("source_snapshot_sha256") != snapshot.get("source_csv_sha256"),
            "response_input_hash_changed": request is not None and result.get("classifier_response", {}).get("Input_SHA256") != request.get("Input_SHA256"),
            "fact_verified_flag": result.get("fact_verified") is not False,
            "backend_changed": batch.get("backend") != NATIVE_BACKEND,
            "result_model_changed": result.get("model") != NATIVE_MODEL,
        }
        stale_reasons = [name for name, failed in stale_checks.items() if failed]
        stale = bool(stale_reasons)
        if stale:
            output[str(bid)] = {"status": "stale_native_context", "backend": NATIVE_BACKEND,
                                "model": NATIVE_MODEL, "batch_id": batch.get("batch_id"),
                                "root_acceptance_sha256": batch.get("root_acceptance_sha256"),
                                "policy_sha256": policy.get("keywords_sha256"),
                                "clarification_sha256": policy.get("clarification_sha256"),
                                "source_csv_sha256": snapshot.get("source_csv_sha256"),
                                "dataset_sha256": batch.get("dataset_sha256"),
                                "stage3_geojson_sha256": snapshot.get("stage3_geojson_sha256"), "response": result,
                                "stale_reasons": stale_reasons}
            continue
        status, main, sub, confidence, adoptable = classify_status(
            result.get("classifier_response", {}), adoption_decision, canonical_main, canonical_sub, taxonomy, confidence_floor)
        if status == "pending_root_adoption" and row_approval is not None:
            try:
                approval_raw = row_approval["approval_utf8"]
                approval = _read_raw_json(approval_raw, "per-record root approval")
                approval_decision = approval.get("decision")
                approval_valid = (sha256_bytes(approval_raw.encode("utf-8")) == row_approval.get("approval_sha256")
                    and approval_decision in ("approved", "rejected")
                    and str(approval.get("building_id")) == str(bid)
                    and approval.get("batch_id") == batch.get("batch_id")
                    and approval.get("context_sha256") == guard.get("context_sha256")
                    and approval.get("root_acceptance_sha256") == batch.get("root_acceptance_sha256")
                    and approval.get("result_sha256") == batch.get("result_sha256")
                    and approval.get("record_binding_sha256") == request.get("record_binding_sha256")
                    and approval.get("model") == NATIVE_MODEL and approval.get("backend") == NATIVE_BACKEND)
                if approval_valid and approval_decision == "rejected":
                    status, main, sub, confidence, adoptable = "reviewed_not_adopted_root_rejected", main, sub, confidence, False
                elif approval_valid:
                    status, main, sub, confidence, adoptable = classify_status(
                        result.get("classifier_response", {}), "root_approved", canonical_main, canonical_sub, taxonomy, confidence_floor)
                else:
                    status = "invalid_root_approval"
            except (KeyError, TypeError, ValueError):
                status = "invalid_root_approval"
        output[str(bid)] = {"status": status, "main": main, "sub": sub, "confidence": confidence,
                            "adoptable": adoptable, "backend": NATIVE_BACKEND, "model": NATIVE_MODEL,
                            "batch_id": batch.get("batch_id"),
                            "root_acceptance_sha256": batch.get("root_acceptance_sha256"),
                            "policy_sha256": policy.get("keywords_sha256"),
                            "clarification_sha256": policy.get("clarification_sha256"),
                            "source_csv_sha256": snapshot.get("source_csv_sha256"),
                            "dataset_sha256": batch.get("dataset_sha256"),
                            "stage3_geojson_sha256": snapshot.get("stage3_geojson_sha256"),
                            "input_sha256": result.get("classifier_response", {}).get("Input_SHA256"),
                            "context_sha256": guard.get("context_sha256"), "response": result}
    for bid in set(batches) - set(str(key) for key in current_rows):
        batch, _, _, response = batches[bid]
        output[bid] = {"status": "stale_native_context", "backend": NATIVE_BACKEND,
                       "model": NATIVE_MODEL, "batch_id": batch.get("batch_id"),
                       "response": response, "reason": "record_absent_from_current_source"}
    return output


