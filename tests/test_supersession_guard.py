import hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from native_function_checkpoint import resolve_occurrences

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def batch(name,bid,input_hash):
    req=json.dumps([{"building_id":bid,"Input_SHA256":input_hash}],separators=(",",":"))
    res=json.dumps([{"building_id":bid,"classifier_response":{"Input_SHA256":input_hash,"LLM_Main_Class":"Unknown_未知类别"}}],ensure_ascii=False,separators=(",",":"))
    return {"batch_id":name,"requests_utf8":req,"request_sha256":sha(req),"results_utf8":res,"result_sha256":sha(res)}
def spec(a,b,prior,current):
    return {"building_id":"7","prior_batch_id":"a","current_batch_id":"b","prior_result_sha256":a["result_sha256"],"current_result_sha256":b["result_sha256"],"prior_input_sha256":prior,"current_input_sha256":current}

class SupersessionTests(unittest.TestCase):
    def setUp(self):
        self.a=batch("a","7","old"); self.b=batch("b","7","new"); self.decl=spec(self.a,self.b,"old","new")
    def test_one_exact_ordered_different_input_replacement(self):
        got=resolve_occurrences([self.a,self.b],[self.decl])
        self.assertEqual(got["7"][0]["batch_id"],"b")
    def test_undeclared_duplicate_rejected(self):
        with self.assertRaises(ValueError): resolve_occurrences([self.a,self.b],[])
    def test_same_input_rejected(self):
        b=batch("b","7","old")
        with self.assertRaisesRegex(ValueError,"same-input"): resolve_occurrences([self.a,b],[spec(self.a,b,"old","old")])
    def test_wrong_result_sha_rejected(self):
        d=dict(self.decl,current_result_sha256="0"*64)
        with self.assertRaisesRegex(ValueError,"result SHA"): resolve_occurrences([self.a,self.b],[d])
    def test_wrong_input_sha_rejected(self):
        d=dict(self.decl,current_input_sha256="x")
        with self.assertRaisesRegex(ValueError,"Input_SHA256"): resolve_occurrences([self.a,self.b],[d])
    def test_reversed_order_rejected(self):
        with self.assertRaisesRegex(ValueError,"order"): resolve_occurrences([self.b,self.a],[self.decl])
    def test_duplicate_override_rejected(self):
        with self.assertRaisesRegex(ValueError,"multiple supersessions"): resolve_occurrences([self.a,self.b],[self.decl,self.decl])
    def test_three_occurrences_rejected(self):
        c=batch("c","7","newer")
        with self.assertRaisesRegex(ValueError,"exactly one"): resolve_occurrences([self.a,self.b,c],[self.decl])
    def test_no_matching_duplicate_declaration_rejected(self):
        with self.assertRaisesRegex(ValueError,"non-duplicate ID"): resolve_occurrences([self.a],[self.decl])
    def test_original_packet_sha_tampering_rejected(self):
        b=dict(self.b); b["results_utf8"]+=' '
        with self.assertRaisesRegex(ValueError,"SHA mismatch"): resolve_occurrences([self.a,b],[self.decl])
    def test_duplicate_result_id_is_rejected_before_mapping(self):
        duplicated=json.loads(self.a["results_utf8"])*2
        self.a["results_utf8"]=json.dumps(duplicated,ensure_ascii=False,separators=(",", ":"))
        self.a["result_sha256"]=sha(self.a["results_utf8"])
        with self.assertRaisesRegex(ValueError,"IDs differ or duplicate"):
            resolve_occurrences([self.a],[])

    def test_input_hash_must_match_response(self):
        b=batch("b","7","new")
        rows=json.loads(b["results_utf8"]); rows[0]["classifier_response"]["Input_SHA256"]="tampered"
        b["results_utf8"]=json.dumps(rows,separators=(",",":")); b["result_sha256"]=sha(b["results_utf8"])
        with self.assertRaisesRegex(ValueError,"does not bind request"): resolve_occurrences([self.a,b],[spec(self.a,b,"old","new")])
if __name__=="__main__": unittest.main()


