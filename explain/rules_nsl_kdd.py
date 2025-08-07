from typing import Dict, Any, List

# -----------------------------
# NSL-KDD full-fledged rules
# -----------------------------
# We cover all common NSL-KDD attack labels and fall back to category rules.
# Thresholds are heuristics based on dataset conventions; tune per your data.
# The rules gracefully skip conditions when a feature is missing.

# Attack -> Category map (covers 39 NSL-KDD labels commonly used)
ATTACK_TO_CAT = {
    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    # normal
    "normal": "normal"
}

# Feature thresholds (tunable)
TH = {
    "short_dur": 1.0,           # seconds
    "very_short_dur": 0.2,
    "high_src_bytes": 5000.0,
    "very_high_src_bytes": 25000.0,
    "low_dst_bytes": 200.0,
    "high_count": 100,
    "high_srv_count": 50,
    "high_serror": 0.5,
    "very_high_serror": 0.7,
    "high_srv_serror": 0.5,
    "wrong_fragment": 1,
    "urgent": 1,
    "hot": 10,
    "num_failed": 3,
    "num_shells": 1,
    "num_access_files": 3,
    "num_file_creations": 2,
    "num_root": 1,
    "su_attempted": 1,
    "root_shell": 1,
    "guest_login": 1,
    "same_srv_rate": 0.7,
    "diff_srv_rate": 0.4,
    "dst_host_srv_count": 100,
    "dst_host_count": 200,
    "shap_min": 0.001  # only claim a feature if |SHAP| >= shap_min
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

# ---- Category-level Templates ----
def explain_dos(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    src_bytes = _val(sample, "src_bytes")
    duration = _val(sample, "duration")
    count = _val(sample, "count")
    srv_count = _val(sample, "srv_count")
    serror = _val(sample, "serror_rate")
    srv_serror = _val(sample, "srv_serror_rate")
    flag = str(sample.get("flag", "")).upper()

    _add_if(phrases,
        (src_bytes > TH["high_src_bytes"] and duration < TH["short_dur"]) and (_shap_ok(shap,"src_bytes") or _shap_ok(shap,"duration")),
        f"High {s(glossary,'src_bytes')} with short {s(glossary,'duration')} is consistent with flooding behavior.")
    _add_if(phrases,
        (serror > TH["high_serror"] or srv_serror > TH["high_srv_serror"]) and (_shap_ok(shap,"serror_rate") or _shap_ok(shap,"srv_serror_rate")),
        f"Elevated {s(glossary,'serror_rate')} / {s(glossary,'srv_serror_rate')} indicates many failed SYN handshakes (DoS signature).")
    _add_if(phrases,
        (count > TH["high_count"] or srv_count > TH["high_srv_count"]) and (_shap_ok(shap,"count") or _shap_ok(shap,"srv_count")),
        f"Unusually high {s(glossary,'count')}/{s(glossary,'srv_count')} suggests repeated rapid connections.")
    _add_if(phrases,
        flag in {"S0","RSTO","RSTR","REJ"} and _present(sample,"flag"),
        f"Connection flag {flag} indicates incomplete or rejected handshakes, common in SYN floods.")
    return phrases

def explain_probe(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    diff_srv_rate = _val(sample, "diff_srv_rate")
    dst_host_srv_count = _val(sample, "dst_host_srv_count")
    dst_host_count = _val(sample, "dst_host_count")
    duration = _val(sample, "duration")

    _add_if(phrases,
        (diff_srv_rate > TH["diff_srv_rate"] or dst_host_srv_count > TH["dst_host_srv_count"] or dst_host_count > TH["dst_host_count"]) and \
            any(_shap_ok(shap,f) for f in ["diff_srv_rate","dst_host_srv_count","dst_host_count"]),
        f"Multiple services/hosts contacted ({s(glossary,'diff_srv_rate')}, {s(glossary,'dst_host_srv_count')}, {s(glossary,'dst_host_count')}) indicate scanning behavior.")
    _add_if(phrases,
        duration < TH["very_short_dur"] and _shap_ok(shap,"duration"),
        f"Very short {s(glossary,'duration')} per connection is typical for probes.")
    return phrases

def explain_r2l(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    logged_in = _val(sample, "logged_in")
    num_failed = _val(sample, "num_failed_logins")
    guest = _val(sample, "is_guest_login")
    hot = _val(sample, "hot")

    _add_if(phrases,
        logged_in == 0 and num_failed >= TH["num_failed"] and any(_shap_ok(shap,f) for f in ["logged_in","num_failed_logins"]),
        f"Repeated failed logins ({s(glossary,'num_failed_logins')}) without a successful login ({s(glossary,'logged_in')}) suggest credential guessing.")
    _add_if(phrases,
        guest == TH["guest_login"] and _shap_ok(shap,"is_guest_login"),
        f"Use of guest login ({s(glossary,'is_guest_login')}) raises risk of unauthorized file operations.")
    _add_if(phrases,
        hot > TH["hot"] and _shap_ok(shap,"hot"),
        f"High {s(glossary,'hot')} (suspicious commands) is consistent with remote-to-local misuse.")
    return phrases

def explain_u2r(sample, shap, glossary) -> List[str]:
    phrases: List[str] = []
    root_shell = _val(sample, "root_shell")
    su_attempted = _val(sample, "su_attempted")
    num_root = _val(sample, "num_root")
    num_shells = _val(sample, "num_shells")
    num_access_files = _val(sample, "num_access_files")
    num_file_creations = _val(sample, "num_file_creations")
    hot = _val(sample, "hot")

    _add_if(phrases,
        (root_shell >= TH["root_shell"] or su_attempted >= TH["su_attempted"] or num_root >= TH["num_root"]) and \
            any(_shap_ok(shap,f) for f in ["root_shell","su_attempted","num_root"]),
        f"Evidence of privilege escalation ({s(glossary,'root_shell')}/{s(glossary,'su_attempted')}/{s(glossary,'num_root')}).")
    _add_if(phrases,
        (num_shells >= TH["num_shells"] or num_access_files >= TH["num_access_files"] or num_file_creations >= TH["num_file_creations"]) and \
            any(_shap_ok(shap,f) for f in ["num_shells","num_access_files","num_file_creations"]),
        f"Suspicious shell/file activity ({s(glossary,'num_shells')}, {s(glossary,'num_access_files')}, {s(glossary,'num_file_creations')}).")
    _add_if(phrases,
        hot > TH["hot"] and _shap_ok(shap,"hot"),
        f"High {s(glossary,'hot')} indicates exploit-like command sequences.")
    return phrases

# ---- Label-specific refinements ----
def label_specializations(label: str, sample: Dict[str,Any], _shap: Dict[str,float], _glossary: Dict[str,str]) -> List[str]:
    L = label.lower()
    phrases: List[str] = []

    # DoS variants
    if L == "neptune":
        serror = _val(sample,"serror_rate"); srv_serror = _val(sample,"srv_serror_rate"); flag = str(sample.get("flag","")).upper()
        if (serror > TH["very_high_serror"] or srv_serror > TH["very_high_serror"]) and (flag in {"S0","RSTO","RSTR","REJ"}):
            phrases.append("SYN flood fingerprint: very high SYN error rates and incomplete TCP handshakes.")
    elif L == "smurf":
        if _eq(sample,"protocol_type","icmp") and _val(sample,"count") > TH["high_count"]:
            phrases.append("ICMP echo broadcast behavior with many rapid requests (classic Smurf DoS).")
    elif L == "pod":
        if (_eq(sample,"protocol_type","icmp") and _val(sample,"src_bytes") > TH["very_high_src_bytes"]) or _val(sample,"wrong_fragment") >= TH["wrong_fragment"]:
            phrases.append("Oversized/fragmented ICMP payload indicative of Ping-of-Death.")
    elif L == "teardrop":
        if _val(sample,"wrong_fragment") >= TH["wrong_fragment"]:
            phrases.append("Malformed IP fragmentation pattern consistent with Teardrop.")
    elif L == "back":
        if _val(sample,"same_srv_rate") > TH["same_srv_rate"] and _val(sample,"count") > TH["high_count"]:
            phrases.append("Back-style backlog saturation against one service (high same-srv-rate and connection count).")
    elif L == "land":
        if _present(sample,"land") and _val(sample,"land") == 1:
            phrases.append("Source and destination IP/port are identical (LAND attack).")
    elif L == "apache2":
        if str(sample.get("service","")).lower() in {"http","www","http_443"} and _val(sample,"srv_count") > TH["high_srv_count"]:
            phrases.append("HTTP service exhaustion (Apache2) via many short requests.")
    elif L == "mailbomb":
        if str(sample.get("service","")).lower() in {"smtp","mail"} and _val(sample,"count") > TH["high_count"]:
            phrases.append("SMTP request burst (mailbomb) causing queue overload.")
    elif L == "udpstorm":
        if _eq(sample,"protocol_type","udp") and _val(sample,"srv_count") > TH["high_srv_count"]:
            phrases.append("UDP flood with many short datagrams (UDP Storm).")
    elif L == "processtable":
        if _val(sample,"srv_count") > TH["high_srv_count"]:
            phrases.append("Service process table exhaustion via excessive connection spawn.")

    # Probe variants
    elif L == "nmap":
        if _val(sample,"diff_srv_rate") > TH["diff_srv_rate"] and _val(sample,"duration") < TH["short_dur"]:
            phrases.append("Service/port sweep with short-lived probes (Nmap-like scanning).")
    elif L == "portsweep":
        if _val(sample,"diff_srv_rate") > TH["diff_srv_rate"] and _val(sample,"dst_host_srv_count") > TH["dst_host_srv_count"]:
            phrases.append("High variety of probed ports across hosts (PortSweep).")
    elif L == "ipsweep":
        if _val(sample,"dst_host_count") > TH["dst_host_count"]:
            phrases.append("Many distinct hosts contacted (IP sweep).")
    elif L == "satan" or L == "mscan" or L == "saint":
        if _val(sample,"diff_srv_rate") > TH["diff_srv_rate"]:
            phrases.append("Automated vulnerability scan across multiple services.")

    # R2L variants
    elif L == "guess_passwd":
        if _val(sample,"num_failed_logins") >= TH["num_failed"]:
            phrases.append("Multiple failed password attempts (guess_passwd).")
    elif L == "ftp_write":
        if str(sample.get("service","")).lower() == "ftp" and _val(sample,"num_file_creations") >= TH["num_file_creations"]:
            phrases.append("Suspicious FTP file write operations.")
    elif L in {"imap","phf","sendmail","named"}:
        svc = str(sample.get("service","")).lower()
        if svc in {L, "smtp" if L=="sendmail" else svc, "domain" if L=="named" else svc}:
            phrases.append(f"Service-targeted R2L ({svc}) activity with anomalous commands.")
    elif L in {"snmpgetattack","snmpguess"}:
        if str(sample.get("service","")).lower() in {"snmp","snmpget"} or _eq(sample,"protocol_type","udp"):
            phrases.append("Suspicious SNMP request patterns (community string guessing/extraction).")
    elif L in {"xlock","xsnoop"}:
        phrases.append("Remote desktop/keylogging attempt indicators (xlock/xsnoop).")
    elif L == "httptunnel":
        if str(sample.get("service","")).lower() in {"http","www"}:
            phrases.append("Data exfiltration through HTTP tunneling heuristics.")

    # U2R variants
    elif L in {"buffer_overflow","loadmodule","perl","rootkit","ps","sqlattack","xterm"}:
        if _val(sample,"root_shell") >= TH["root_shell"] or _val(sample,"num_root") >= TH["num_root"] or _val(sample,"su_attempted") >= TH["su_attempted"]:
            phrases.append("Exploit culminating in root privileges (U2R hallmark).")
        if _val(sample,"hot") > TH["hot"]:
            phrases.append("Exploit-like command sequence density (high 'hot').")

    return phrases

def explain_instance(sample: Dict[str, Any], pred_label: str, shap_dict: Dict[str, float], glossary: Dict[str,str]) -> str:
    """Return a human-readable explanation grounded in rules + SHAP."""
    label = (pred_label or "").lower()
    cat = ATTACK_TO_CAT.get(label, sample.get("attack_cat", "") or "unknown")
    cat = cat if isinstance(cat, str) else str(cat)
    cat = cat if cat else "unknown"

    pieces: List[str] = []
    # Category scaffolding
    if cat == "DoS":
        pieces += explain_dos(sample, shap_dict, glossary)
    elif cat == "Probe":
        pieces += explain_probe(sample, shap_dict, glossary)
    elif cat == "R2L":
        pieces += explain_r2l(sample, shap_dict, glossary)
    elif cat == "U2R":
        pieces += explain_u2r(sample, shap_dict, glossary)

    # Label-specific refinement
    pieces += label_specializations(label, sample, shap_dict, glossary)

    # SHAP top factors (mention the strongest drivers)
    # We insert up to 5 strongest features by |SHAP|
    tops = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    shap_desc = []
    for f, v in tops:
        if abs(v) >= TH["shap_min"]:
            direction = "increased" if v > 0 else "decreased"
            shap_desc.append(f"{s(glossary,f)} {direction} the model's confidence")

    base = f"The model classified this connection as '{pred_label}' (category: {cat})."
    details = (" " + " ".join(pieces)) if pieces else " Behavior matches the predicted category's typical indicators."
    shap_txt = (" Key factors: " + "; ".join(shap_desc) + ".") if shap_desc else ""
    return base + details + shap_txt
