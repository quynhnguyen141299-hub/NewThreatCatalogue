"""
siem.py — SIEM-based Security Controls Automation Engine

For each flagged threat, the engine:
1. Matches STRIDE tags + ASAP layer + severity to a rule
2. Fires automated actions (contain, alert, escalate, recover)
3. Returns a full incident report per transaction
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime, timezone
import ast
import uuid


# ─────────────────────────────────────────────
# SIEM Action Types
# ─────────────────────────────────────────────

class ActionType:
    DETECT  = "DETECT"
    PROTECT = "PROTECT"
    RESPOND = "RESPOND"
    RECOVER = "RECOVER"
    LOG     = "LOG"
    ESCALATE= "ESCALATE"


@dataclass
class SiemAction:
    action_type:  str
    nist_control: str
    description:  str
    automated:    bool = True   # True = auto-executed, False = requires human


@dataclass
class SiemIncident:
    incident_id:   str
    transaction_id: int
    timestamp:     str
    severity:      str
    stride_tags:   List[str]
    asap_layer:    str
    mitre_tactics: str
    mitre_techniques: str
    nist_controls: str
    triggered_rules: List[str]
    actions:       List[SiemAction]
    escalate:      bool
    status:        str   # "AUTO_CONTAINED", "ESCALATED", "MONITORING", "RECOVERING"


# ─────────────────────────────────────────────
# Rule definitions
# ─────────────────────────────────────────────

# Each rule: (stride_tag, asap_layer, min_severity) → list of SiemActions
# min_severity: 1=Low, 2=Medium, 3=High

SIEM_RULES: List[Dict[str, Any]] = [

    # ── Spoofing ──────────────────────────────────────────────────────
    {
        "rule_id":     "SIEM-001",
        "name":        "Credential Spoofing — Access Layer",
        "stride":      "Spoofing",
        "layer":       "Access",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.AE-2",          "Raise alert: Credential Spoofing Attempt detected",             True),
            SiemAction(ActionType.PROTECT, "PR.AC-1",          "Revoke session token for affected wallet_id",                   True),
            SiemAction(ActionType.PROTECT, "PR.AC-3",          "Trigger MFA re-authentication challenge",                       True),
            SiemAction(ActionType.RESPOND, "RS.MI-1",          "Block source IP for 30 minutes",                                True),
            SiemAction(ActionType.LOG,     "DE.CM-3",          "Write incident to audit log D4_LOGS with full context",         True),
        ],
    },
    {
        "rule_id":     "SIEM-002",
        "name":        "KYC Forgery — Access Layer",
        "stride":      "Spoofing",
        "layer":       "Access",
        "min_votes":   2,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.AE-2",          "Raise critical alert: KYC Forgery Attempt",                     True),
            SiemAction(ActionType.PROTECT, "PR.AC-4",          "Suspend onboarding for affected user_id",                       True),
            SiemAction(ActionType.RESPOND, "RS.MI-1",          "Flag user_id for manual KYC re-verification",                   False),
            SiemAction(ActionType.ESCALATE,"RS.CO-2",          "Notify compliance team immediately",                            False),
        ],
    },

    # ── Tampering ─────────────────────────────────────────────────────
    {
        "rule_id":     "SIEM-003",
        "name":        "Payment Tampering — Service Layer",
        "stride":      "Tampering",
        "layer":       "Service",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.CM-3",          "Raise alert: Payment Tampering Detected",                       True),
            SiemAction(ActionType.PROTECT, "PR.DS-6",          "Verify transaction integrity hash",                             True),
            SiemAction(ActionType.RESPOND, "RS.AN-1",          "Queue transaction for analyst investigation",                   True),
            SiemAction(ActionType.LOG,     "DE.CM-3",          "Log anomalous transaction features to SIEM",                    True),
        ],
    },
    {
        "rule_id":     "SIEM-004",
        "name":        "Ledger Tampering — Asset Layer",
        "stride":      "Tampering",
        "layer":       "Asset",
        "min_votes":   2,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.AE-4",          "Raise critical alert: Ledger Tampering Detected",               True),
            SiemAction(ActionType.PROTECT, "PR.DS-1",          "Freeze affected wallet_id immediately",                         True),
            SiemAction(ActionType.PROTECT, "PR.DS-6",          "Snapshot ledger state for forensic integrity",                  True),
            SiemAction(ActionType.RESPOND, "RS.AN-1",          "Escalate to senior analyst queue",                              True),
            SiemAction(ActionType.RECOVER, "RC.RP-1",          "Trigger ledger rollback to last verified checkpoint",           False),
            SiemAction(ActionType.ESCALATE,"RS.CO-2",          "Notify central bank operations team",                           False),
        ],
    },
    {
        "rule_id":     "SIEM-005",
        "name":        "Platform Config Tampering",
        "stride":      "Tampering",
        "layer":       "Platform",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.AE-3",          "Raise alert: Unauthorised Config Change",                       True),
            SiemAction(ActionType.PROTECT, "PR.IP-1",          "Revert config to last known good baseline",                     True),
            SiemAction(ActionType.RESPOND, "RS.AN-1",          "Investigate source of config change",                           True),
            SiemAction(ActionType.LOG,     "DE.CM-3",          "Log config delta with before/after values",                     True),
        ],
    },

    # ── Information Disclosure ────────────────────────────────────────
    {
        "rule_id":     "SIEM-006",
        "name":        "Data Exfiltration — Asset Layer",
        "stride":      "InformationDisclosure",
        "layer":       "Asset",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.CM-7",          "Raise alert: Sensitive Ledger Access Anomaly",                  True),
            SiemAction(ActionType.PROTECT, "PR.DS-5",          "Mask sensitive fields in subsequent API responses",             True),
            SiemAction(ActionType.RESPOND, "RS.AN-3",          "Quarantine agent_id for investigation",                         True),
            SiemAction(ActionType.RECOVER, "RC.IM-1",          "Review and tighten data access policies",                       False),
        ],
    },
    {
        "rule_id":     "SIEM-007",
        "name":        "Balance Query Anomaly — Service Layer",
        "stride":      "InformationDisclosure",
        "layer":       "Service",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.CM-1",          "Raise alert: Anomalous Balance Query Pattern",                  True),
            SiemAction(ActionType.PROTECT, "PR.AC-4",          "Apply query rate limiting for agent_id",                        True),
            SiemAction(ActionType.LOG,     "DE.CM-7",          "Cross-correlate with other queries from same agent_id",         True),
        ],
    },

    # ── Denial of Service ─────────────────────────────────────────────
    {
        "rule_id":     "SIEM-008",
        "name":        "Resource Exhaustion — Platform Layer",
        "stride":      "DenialOfService",
        "layer":       "Platform",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.AE-5",          "Raise alert: Resource Exhaustion / DoS Detected",               True),
            SiemAction(ActionType.PROTECT, "PR.PT-4",          "Apply rate limiting on affected endpoint",                      True),
            SiemAction(ActionType.RESPOND, "RS.RP-1",          "Execute DoS response playbook",                                 True),
            SiemAction(ActionType.RECOVER, "RC.RP-1",          "Auto-scale platform resources to restore availability",         True),
            SiemAction(ActionType.LOG,     "DE.CM-1",          "Record attack complexity, duration and source metrics",         True),
        ],
    },

    # ── Repudiation ───────────────────────────────────────────────────
    {
        "rule_id":     "SIEM-009",
        "name":        "Transaction Replay / Repudiation — Service Layer",
        "stride":      "Repudiation",
        "layer":       "Service",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.CM-3",          "Raise alert: Transaction Replay Attempt Detected",              True),
            SiemAction(ActionType.PROTECT, "PR.IP-6",          "Invalidate duplicate transaction ID",                           True),
            SiemAction(ActionType.RESPOND, "RS.CO-2",          "Report incident to audit and compliance systems",               True),
            SiemAction(ActionType.RECOVER, "RC.IM-2",          "Update anti-replay nonce policy",                               False),
        ],
    },
    {
        "rule_id":     "SIEM-010",
        "name":        "Offline Double Spend — Access Layer",
        "stride":      "Repudiation",
        "layer":       "Access",
        "min_votes":   2,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.CM-3",          "Raise critical alert: Offline Double Spend Detected",           True),
            SiemAction(ActionType.PROTECT, "PR.DS-1",          "Hold offline transaction pending reconciliation",                True),
            SiemAction(ActionType.RESPOND, "RS.AN-1",          "Trigger offline buffer audit",                                  True),
            SiemAction(ActionType.ESCALATE,"RS.CO-2",          "Alert settlement team for manual reconciliation",               False),
            SiemAction(ActionType.RECOVER, "RC.RP-1",          "Revert double-spent balance to last confirmed state",           False),
        ],
    },

    # ── Elevation of Privilege ────────────────────────────────────────
    {
        "rule_id":     "SIEM-011",
        "name":        "Privilege Escalation — Platform Layer",
        "stride":      "ElevationOfPrivilege",
        "layer":       "Platform",
        "min_votes":   1,
        "actions": [
            SiemAction(ActionType.DETECT,  "DE.AE-1",          "Raise alert: Privilege Escalation Attempt",                     True),
            SiemAction(ActionType.PROTECT, "PR.AC-6",          "Demote user permissions to least privilege",                    True),
            SiemAction(ActionType.PROTECT, "PR.IP-1",          "Revert platform config to baseline",                            True),
            SiemAction(ActionType.RESPOND, "RS.MI-3",          "Terminate active session for agent_id",                         True),
            SiemAction(ActionType.RESPOND, "RS.MI-3",          "Lock account pending security review",                          True),
            SiemAction(ActionType.RECOVER, "RC.RP-1",          "Restore access controls to known good state",                   False),
            SiemAction(ActionType.ESCALATE,"RS.CO-2",          "Notify security operations centre",                             False),
        ],
    },
]


# ─────────────────────────────────────────────
# SIEM Engine
# ─────────────────────────────────────────────

class SiemEngine:

    def __init__(self):
        self.rules     = SIEM_RULES
        self.incidents: List[SiemIncident] = []

    def _parse_tags(self, stride_tags_raw: str) -> List[str]:
        try:
            return ast.literal_eval(stride_tags_raw) if isinstance(stride_tags_raw, str) else []
        except:
            return []

    def _severity_int(self, severity: str) -> int:
        return {"Low": 1, "Medium": 2, "High": 3}.get(severity, 1)

    def _match_rules(self, tags: List[str], layer: str, votes: int) -> List[Dict]:
        matched = []
        for rule in self.rules:
            if rule["stride"] in tags and rule["layer"] == layer and votes >= rule["min_votes"]:
                matched.append(rule)
        return matched

    def _determine_status(self, severity: str, actions: List[SiemAction]) -> str:
        has_escalate = any(a.action_type == ActionType.ESCALATE for a in actions)
        has_recover  = any(a.action_type == ActionType.RECOVER  for a in actions)
        if severity == "High" and has_escalate:
            return "ESCALATED"
        elif has_recover:
            return "RECOVERING"
        elif severity == "Low":
            return "MONITORING"
        else:
            return "AUTO_CONTAINED"

    def process(self, row: Dict[str, Any]) -> SiemIncident:
        # Support both raw column names and mapped catalogue column names
        stride_raw = row.get("STRIDE Tags", row.get("stride_tags", ""))
        layer_raw  = row.get("ASAP Layer",  row.get("asap_layer",  ""))

        # STRIDE Tags in catalogue are comma-separated strings not JSON lists
        if isinstance(stride_raw, str) and stride_raw.startswith("["):
            tags = self._parse_tags(stride_raw)
        elif isinstance(stride_raw, str):
            tags = [t.strip() for t in stride_raw.split(",") if t.strip()]
        else:
            tags = []

        layer    = str(layer_raw).strip()
        severity = str(row.get("Severity", "Low"))
        votes    = {"High": 3, "Medium": 2, "Low": 1}.get(severity, 1)
        tx_id    = int(row.get("Transaction #", 0))

        matched_rules  = self._match_rules(tags, layer, votes)
        triggered_names= [r["rule_id"] + ": " + r["name"] for r in matched_rules]
        all_actions    = []
        for rule in matched_rules:
            all_actions.extend(rule["actions"])

        # Deduplicate actions by description
        seen, unique_actions = set(), []
        for a in all_actions:
            if a.description not in seen:
                seen.add(a.description)
                unique_actions.append(a)

        # Escalate automatically for High severity
        escalate = severity == "High" or any(a.action_type == ActionType.ESCALATE for a in unique_actions)
        status   = self._determine_status(severity, unique_actions)

        incident = SiemIncident(
            incident_id      = f"INC-{str(uuid.uuid4())[:8].upper()}",
            transaction_id   = tx_id,
            timestamp        = datetime.now(timezone.utc).isoformat(),
            severity         = severity,
            stride_tags      = tags,
            asap_layer       = layer,
            mitre_tactics    = row.get("MITRE Tactics",    ""),
            mitre_techniques = row.get("MITRE Techniques", ""),
            nist_controls    = row.get("NIST Controls",    ""),
            triggered_rules  = triggered_names,
            actions          = unique_actions,
            escalate         = escalate,
            status           = status,
        )
        self.incidents.append(incident)
        return incident

    def process_all(self, catalogue_df) -> List[SiemIncident]:
        self.incidents = []
        for _, row in catalogue_df.iterrows():
            self.incidents.append(self.process(row.to_dict()))
        return self.incidents

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        for inc in self.incidents:
            auto_actions   = [a.description for a in inc.actions if a.automated]
            manual_actions = [a.description for a in inc.actions if not a.automated]
            rows.append({
                "Incident ID":       inc.incident_id,
                "Transaction #":     inc.transaction_id,
                "Timestamp":         inc.timestamp,
                "Severity":          inc.severity,
                "Status":            inc.status,
                "STRIDE Tags":       ", ".join(inc.stride_tags),
                "ASAP Layer":        inc.asap_layer,
                "MITRE Techniques":  inc.mitre_techniques,
                "NIST Controls":     inc.nist_controls,
                "Triggered Rules":   " | ".join(inc.triggered_rules),
                "Auto Actions":      " → ".join(auto_actions),
                "Manual Actions":    " → ".join(manual_actions),
                "Escalate":          "Yes" if inc.escalate else "No",
            })
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        if not self.incidents:
            return {}
        statuses  = [i.status   for i in self.incidents]
        severities= [i.severity for i in self.incidents]
        import pandas as pd
        return {
            "Total Incidents":   len(self.incidents),
            "Auto Contained":    statuses.count("AUTO_CONTAINED"),
            "Escalated":         statuses.count("ESCALATED"),
            "Recovering":        statuses.count("RECOVERING"),
            "Monitoring":        statuses.count("MONITORING"),
            "High":              severities.count("High"),
            "Medium":            severities.count("Medium"),
            "Low":               severities.count("Low"),
            "Rules Fired":       len(set(r for i in self.incidents for r in i.triggered_rules)),
        }