from typing import Dict, Any, List

# -----------------------------
# UNSW-NB15 full-fledged rules
# -----------------------------
# Uses attack categories: Fuzzers, Analysis, Backdoor, DoS, Exploits,
# Generic, Reconnaissance, Shellcode, Worms, plus Normal.
# Thresholds are heuristic; tune to your distribution.

TH = {
    "short_dur": 1.0,              # seconds
    "very_short_dur": 0.2,
    "high_sbytes": 1e5,
    "vhigh_sbytes": 5e5,
    "low_dbytes": 1e3,
    "high_spkts": 800,
    "high_dpkts": 800,
    "high_sload": 1e5,             # bytes/s approx
    "high_dload": 1e5,
    "symmetry_tol": 0.25,          # |sbytes-dbytes|/max ~<= 0.25
    "high_ct_dst_ltm": 100,
    "high_ct_srv_dst": 100,
    "high_ct_src_dport_ltm": 50,
    "high_ct_state_ttl": 20,
    "high_rate": 2e5,              # packets/s or bytes/s depending on feature
    "shap_min": 0.001
}

def s(glossary: Dict[str,str], f: str) -> str:
    return glossary.get(f, f)

def _val(sample: Dict[str, Any], key: str, default=0.0):
    v = sample.get(key, default)
    try:
        return float(v) if isinstance(v, (int, float, str)) and str(v).strip() != "" else default
    except Exception:
        return default

def _eq(sample: Dict[str, Any], key: str, value) -> bool:
    return str(sample.get(key, "")).lower() == str(value).lower()

def _present(sample: Dict[str, Any], key: str) -> bool:
    return key in sample

def _shap_ok(shap: Dict[str, float], feat: str) -> bool:
    return abs(shap.get(feat, 0.0)) >= TH["shap_min"]

def _add_if(phrases: List[str], cond: bool, text: str):
    if cond:
        phrases.append(text)

def _rough_symmetry(a: float, b: float, tol: float) -> bool:
    m = max(a, b, 1.0)
    return abs(a - b)/m <= tol

# ---- Category templates ----
def explain_dos(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    sbytes = _val(sample,"sbytes")
    spkts  = _val(sample,"spkts")
    dur    = _val(sample,"dur");   state  = str(sample.get("state","")).upper()
    sload  = _val(sample,"sload")
    _add_if(phrases,
        (sbytes > TH["high_sbytes"] and dur < TH["short_dur"]) and (_shap_ok(shap,"sbytes") or _shap_ok(shap,"dur")),
        f"Large {s(glossary,'sbytes')} in short {s(glossary,'dur')} suggests flooding.")
    _add_if(phrases,
        (spkts > TH["high_spkts"]) and _shap_ok(shap,"spkts"),
        f"High {s(glossary,'spkts')} consistent with packet flood.")
    _add_if(phrases,
        state in {"RST","RSTO","REJ"} and _present(sample,"state"),
        f"Abnormal TCP termination state {state} during bursts.")
    _add_if(phrases,
        sload > TH["high_sload"] and _shap_ok(shap,"sload"),
        f"Elevated sender load ({s(glossary,'sload')}).")
    return phrases

def explain_recon(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    ct_dst_ltm = _val(sample,"ct_dst_ltm")
    ct_srv_dst = _val(sample,"ct_srv_dst")
    dur = _val(sample,"dur")
    _add_if(phrases,
        (ct_dst_ltm > TH["high_ct_dst_ltm"] or ct_srv_dst > TH["high_ct_srv_dst"]) and \
            ( _shap_ok(shap,"ct_dst_ltm") or _shap_ok(shap,"ct_srv_dst") ),
        f"Many distinct destinations/services contacted ({s(glossary,'ct_dst_ltm')}, {s(glossary,'ct_srv_dst')}).")
    _add_if(phrases,
        dur < TH["very_short_dur"] and _shap_ok(shap,"dur"),
        "Very short-lived connections typical of probing.")
    return phrases

def explain_exploits(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    dur = _val(sample,"dur"); sbytes = _val(sample,"sbytes"); dbytes = _val(sample,"dbytes")
    _add_if(phrases,
        (dur < TH["short_dur"] and sbytes > TH["high_sbytes"] and dbytes < TH["low_dbytes"]) and \
            any(_shap_ok(shap,f) for f in ["dur","sbytes","dbytes"]),
        f"Short exploit attempt with large outbound bytes and little response.")
    return phrases

def explain_generic(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    sbytes = _val(sample,"sbytes"); dbytes = _val(sample,"dbytes")
    sload = _val(sample,"sload"); dload = _val(sample,"dload")
    _add_if(phrases,
        (sload > TH["high_sload"] and dload > TH["high_dload"]) and any(_shap_ok(shap,f) for f in ["sload","dload"]),
        f"High bidirectional throughput ({s(glossary,'sload')}, {s(glossary,'dload')}).")
    _add_if(phrases,
        _rough_symmetry(sbytes,dbytes, TH["symmetry_tol"]) and ( _shap_ok(shap,"sbytes") or _shap_ok(shap,"dbytes") ),
        f"Near-symmetric byte volumes; signature-like bulk transfer (Generic).")
    return phrases

def explain_fuzzers(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    ct_src_dport_ltm = _val(sample,"ct_src_dport_ltm")
    dloss = _val(sample,"dloss"); sloss = _val(sample,"sloss")
    _add_if(phrases,
        (ct_src_dport_ltm > TH["high_ct_src_dport_ltm"]) and _shap_ok(shap,"ct_src_dport_ltm"),
        f"High variety of destination ports per source ({s(glossary,'ct_src_dport_ltm')}) indicates fuzzing attempts.")
    _add_if(phrases,
        (dloss > 0 or sloss > 0) and ( _shap_ok(shap,"dloss") or _shap_ok(shap,"sloss") ),
        f"Packet loss while exploring inputs ({s(glossary,'dloss')}/{s(glossary,'sloss')}).")
    return phrases

def explain_backdoor(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    is_sm = _val(sample,"is_sm_ips_ports")
    dur = _val(sample,"dur"); dbytes = _val(sample,"dbytes")
    _add_if(phrases,
        is_sm == 1 and _shap_ok(shap,"is_sm_ips_ports"),
        f"Same IPs and ports across flows ({s(glossary,'is_sm_ips_ports')}) suggests fixed backdoor channel.")
    _add_if(phrases,
        (dur > 10 and dbytes > TH['low_dbytes']) and any(_shap_ok(shap,f) for f in ["dur","dbytes"]),
        f"Long-lived session with steady response volume characteristic of remote access.")
    return phrases

def explain_shellcode(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    dttl = _val(sample,"dttl"); sttl = _val(sample,"sttl")
    ct_state_ttl = _val(sample,"ct_state_ttl")
    _add_if(phrases,
        (ct_state_ttl > TH["high_ct_state_ttl"]) and _shap_ok(shap,"ct_state_ttl"),
        f"Unusual combinations of state/TTL across flows ({s(glossary,'ct_state_ttl')}).")
    _add_if(phrases,
        (dttl < 32 or sttl < 32) and ( _shap_ok(shap,"dttl") or _shap_ok(shap,"sttl") ),
        f"Low TTL values can accompany shellcode delivery attempts.")
    return phrases

def explain_analysis(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    svc = str(sample.get("service","")).lower()
    http_m = _val(sample,"ct_flw_http_mthd")
    _add_if(phrases,
        svc in {"dns","http","ftp","smtp"} and _present(sample,"service"),
        f"Traffic is focused on analysis-prone services ({svc}).")
    _add_if(phrases,
        (http_m > 0) and _shap_ok(shap,"ct_flw_http_mthd"),
        f"Presence of HTTP methods across flows ({s(glossary,'ct_flw_http_mthd')}).")
    return phrases

def explain_worms(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    ct_dst_ltm = _val(sample,"ct_dst_ltm"); ct_srv_dst = _val(sample,"ct_srv_dst")
    sbytes = _val(sample,"sbytes"); dbytes = _val(sample,"dbytes")
    _add_if(phrases,
        (ct_dst_ltm > TH["high_ct_dst_ltm"] and ct_srv_dst > TH["high_ct_srv_dst"]) and \
            ( _shap_ok(shap,"ct_dst_ltm") or _shap_ok(shap,"ct_srv_dst") ),
        f"Rapid propagation pattern across many hosts and services.")
    _add_if(phrases,
        (sbytes > TH["high_sbytes"] and dbytes < TH["low_dbytes"]) and any(_shap_ok(shap,f) for f in ["sbytes","dbytes"]),
        f"High outbound with minimal response, consistent with worm scanning/infection.")
    return phrases

# Router
CATEGORY_RULES = {
    "DoS": explain_dos,
    "Reconnaissance": explain_recon,
    "Explosits": explain_exploits,   # (typo-proofing) keep mapping too
    "Exploits": explain_exploits,
    "Generic": explain_generic,
    "Fuzzers": explain_fuzzers,
    "Backdoor": explain_backdoor,
    "Shellcode": explain_shellcode,
    "Analysis": explain_analysis,
    "Worms": explain_worms,
    "Normal": lambda *args, **kwargs: ["Benign traffic characteristics."]
}

def explain_instance(sample: Dict[str, Any], pred_label: str, shap_dict: Dict[str, float], glossary: Dict[str,str]) -> str:
    # pred_label expected to be one of the UNSW-NB15 categories (or 'Normal')
    cat = (pred_label or sample.get("attack_cat","") or "Unknown").title()
    rule_fn = CATEGORY_RULES.get(cat, CATEGORY_RULES.get(pred_label, None))
    phrases: List[str] = []
    hit = 0
    if rule_fn is not None:
        phrases += rule_fn(sample, shap_dict, glossary)
        if phrases:
            hit = 1
    else:
        phrases.append("Heuristics for this category are not defined; using generic SHAP rationale.")

    # SHAP mention
    tops = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    shap_desc = []
    for f, v in tops:
        if abs(v) >= TH["shap_min"]:
            direction = "increased" if v > 0 else "decreased"
            shap_desc.append(f"{s(glossary,f)} {direction} the model's confidence")

    base = f"The model classified this flow as '{pred_label}'."
    details = (" " + " ".join(phrases)) if phrases else ""
    shap_txt = (" Key factors: " + "; ".join(shap_desc) + ".") if shap_desc else ""
    return base + details + shap_txt, hit

