import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import IsolationForest
import ast

from preprocess import preprocess
from detector import detect
from siem import SiemEngine, ActionType

st.set_page_config(layout="wide", page_title="CBDC Threat Detection")

# ─────────────────────────────────────────────
# MITRE ATT&CK Mapping
# ─────────────────────────────────────────────
STRIDE_TO_MITRE = {
    "Spoofing":             {"tactic":"Initial Access",      "tactic_id":"TA0001","technique":"Valid Accounts",                       "technique_id":"T1078","description":"Adversary uses stolen or forged credentials to gain access."},
    "Tampering":            {"tactic":"Impact",              "tactic_id":"TA0040","technique":"Data Manipulation",                    "technique_id":"T1565","description":"Adversary manipulates data to disrupt availability or integrity."},
    "InformationDisclosure":{"tactic":"Collection",          "tactic_id":"TA0009","technique":"Data from Information Repositories",   "technique_id":"T1213","description":"Adversary accesses sensitive data from repositories."},
    "DenialOfService":      {"tactic":"Impact",              "tactic_id":"TA0040","technique":"Network Denial of Service",            "technique_id":"T1498","description":"Adversary disrupts availability of network services."},
    "Repudiation":          {"tactic":"Defense Evasion",     "tactic_id":"TA0005","technique":"Indicator Removal",                    "technique_id":"T1070","description":"Adversary removes or alters logs to deny actions taken."},
    "ElevationOfPrivilege": {"tactic":"Privilege Escalation","tactic_id":"TA0004","technique":"Exploitation for Privilege Escalation","technique_id":"T1068","description":"Adversary exploits a vulnerability to gain elevated privileges."},
}
LAYER_TO_MITRE = {
    "Access":   {"tactic":"Initial Access",      "tactic_id":"TA0001","technique":"Exploit Public-Facing Application",     "technique_id":"T1190"},
    "Service":  {"tactic":"Execution",           "tactic_id":"TA0002","technique":"Exploitation for Client Execution",     "technique_id":"T1203"},
    "Asset":    {"tactic":"Impact",              "tactic_id":"TA0040","technique":"Financial Theft",                       "technique_id":"T1657"},
    "Platform": {"tactic":"Privilege Escalation","tactic_id":"TA0004","technique":"Exploitation for Privilege Escalation", "technique_id":"T1068"},
}

# ─────────────────────────────────────────────
# NIST CSF Mapping
# ─────────────────────────────────────────────
STRIDE_TO_NIST = {
    "Spoofing": [
        {"function":"Protect","category":"Identity Management & Access Control","controls":"PR.AC-1, PR.AC-3","description":"Manage identities and credentials; control remote access."},
        {"function":"Protect","category":"Protective Technology",               "controls":"PR.PT-3",          "description":"Apply principle of least functionality."},
        {"function":"Detect", "category":"Anomalies and Events",                "controls":"DE.AE-2",          "description":"Analyse detected spoofing events and their impact."},
        {"function":"Respond","category":"Mitigation",                          "controls":"RS.MI-1",          "description":"Contain spoofing incidents to limit impact."},
    ],
    "Tampering": [
        {"function":"Protect","category":"Data Security",                       "controls":"PR.DS-1, PR.DS-6", "description":"Protect data at rest; verify data integrity."},
        {"function":"Detect", "category":"Security Continuous Monitoring",      "controls":"DE.CM-3",          "description":"Monitor personnel activity for tampering anomalies."},
        {"function":"Respond","category":"Analysis",                            "controls":"RS.AN-1",          "description":"Investigate detected tampering events thoroughly."},
        {"function":"Recover","category":"Recovery Planning",                   "controls":"RC.RP-1",          "description":"Restore integrity of tampered data and systems."},
    ],
    "InformationDisclosure": [
        {"function":"Protect","category":"Data Security",                       "controls":"PR.DS-5",          "description":"Implement protections against data leaks."},
        {"function":"Protect","category":"Access Control",                      "controls":"PR.AC-4",          "description":"Manage access permissions and authorisations."},
        {"function":"Detect", "category":"Security Continuous Monitoring",      "controls":"DE.CM-7",          "description":"Monitor for unauthorised data access and disclosure."},
        {"function":"Respond","category":"Analysis",                            "controls":"RS.AN-3",          "description":"Investigate data disclosure incidents and scope."},
        {"function":"Recover","category":"Improvements",                        "controls":"RC.IM-1",          "description":"Incorporate lessons learned from disclosure events."},
    ],
    "DenialOfService": [
        {"function":"Protect","category":"Protective Technology",               "controls":"PR.PT-4",          "description":"Protect communications and control networks."},
        {"function":"Detect", "category":"Anomalies and Events",                "controls":"DE.AE-5",          "description":"Establish alert thresholds for DoS activity."},
        {"function":"Respond","category":"Response Planning",                   "controls":"RS.RP-1",          "description":"Execute response plan during or after a DoS event."},
        {"function":"Recover","category":"Recovery Planning",                   "controls":"RC.RP-1",          "description":"Restore availability of services after DoS."},
    ],
    "Repudiation": [
        {"function":"Protect","category":"Information Protection Processes",    "controls":"PR.IP-6",          "description":"Destroy data according to policy; maintain audit logs."},
        {"function":"Detect", "category":"Security Continuous Monitoring",      "controls":"DE.CM-3, DE.CM-7", "description":"Monitor for unauthorised activity and personnel actions."},
        {"function":"Respond","category":"Communications",                      "controls":"RS.CO-2",          "description":"Report repudiation incidents to appropriate parties."},
        {"function":"Recover","category":"Improvements",                        "controls":"RC.IM-2",          "description":"Update strategies based on repudiation lessons learned."},
    ],
    "ElevationOfPrivilege": [
        {"function":"Protect","category":"Identity Management & Access Control","controls":"PR.AC-4, PR.AC-6", "description":"Manage access permissions; use principle of least privilege."},
        {"function":"Detect", "category":"Anomalies and Events",                "controls":"DE.AE-1",          "description":"Establish baseline; detect privilege escalation attempts."},
        {"function":"Respond","category":"Mitigation",                          "controls":"RS.MI-3",          "description":"Neutralise privilege escalation and revoke access."},
        {"function":"Recover","category":"Recovery Planning",                   "controls":"RC.RP-1",          "description":"Restore access controls to known good state."},
    ],
}
LAYER_TO_NIST = {
    "Access":  [
        {"function":"Identify","category":"Asset Management",                    "controls":"ID.AM-1",          "description":"Inventory access points and authentication assets."},
        {"function":"Protect", "category":"Identity Management & Access Control","controls":"PR.AC-1, PR.AC-7", "description":"Manage credentials; authenticate users and devices."},
        {"function":"Detect",  "category":"Security Continuous Monitoring",      "controls":"DE.CM-1",          "description":"Monitor access layer for anomalous activity."},
        {"function":"Respond", "category":"Mitigation",                          "controls":"RS.MI-2",          "description":"Block unauthorised access attempts at entry points."},
    ],
    "Service": [
        {"function":"Identify","category":"Risk Assessment",                     "controls":"ID.RA-1",          "description":"Identify vulnerabilities in service layer components."},
        {"function":"Protect", "category":"Protective Technology",               "controls":"PR.PT-1",          "description":"Audit and log service layer activity."},
        {"function":"Detect",  "category":"Security Continuous Monitoring",      "controls":"DE.CM-1, DE.CM-8", "description":"Monitor network and detect service vulnerabilities."},
        {"function":"Respond", "category":"Analysis",                            "controls":"RS.AN-2",          "description":"Understand the impact of service layer incidents."},
    ],
    "Asset":   [
        {"function":"Identify","category":"Asset Management",                    "controls":"ID.AM-1, ID.AM-2", "description":"Inventory physical and software assets."},
        {"function":"Protect", "category":"Data Security",                       "controls":"PR.DS-1, PR.DS-2", "description":"Protect data at rest and in transit."},
        {"function":"Detect",  "category":"Anomalies and Events",                "controls":"DE.AE-4",          "description":"Determine impact of asset-level events."},
        {"function":"Recover", "category":"Asset Recovery",                      "controls":"RC.RP-1",          "description":"Restore and recover compromised assets."},
    ],
    "Platform":[
        {"function":"Identify","category":"Business Environment",                "controls":"ID.BE-5",          "description":"Establish resilience requirements for platform services."},
        {"function":"Protect", "category":"Information Protection Processes",    "controls":"PR.IP-1, PR.IP-3", "description":"Maintain baseline configurations; manage configuration change."},
        {"function":"Detect",  "category":"Anomalies and Events",                "controls":"DE.AE-3",          "description":"Aggregate and correlate platform event data."},
        {"function":"Respond", "category":"Response Planning",                   "controls":"RS.RP-1",          "description":"Execute platform-level incident response plan."},
        {"function":"Recover", "category":"Recovery Planning",                   "controls":"RC.RP-1",          "description":"Restore platform services to operational state."},
    ],
}
NIST_FUNCTION_COLORS = {
    "Identify":"#4C72B0","Protect":"#55A868","Detect":"#DD8452",
    "Respond":"#C44E52","Recover":"#8172B2",
}
STATUS_COLORS = {
    "ESCALATED":      "#5c1a1a",
    "AUTO_CONTAINED": "#1a3a5c",
    "RECOVERING":     "#3a1a5c",
    "MONITORING":     "#1a3a1a",
}

# Severity-based priority NIST function — keeps only the most actionable
# controls for a given severity level instead of showing all 5 functions.
SEVERITY_TO_PRIORITY_FUNCTION = {
    "Low":    "Detect",
    "Medium": "Protect",
    "High":   "Respond",
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_severity(v):
    try: v=int(float(v))
    except: v=0
    return "High" if v>=3 else "Medium" if v==2 else "Low"

def map_to_catalogue(row):
    try: tags=ast.literal_eval(row["stride_tags"]) if isinstance(row["stride_tags"],str) else []
    except: tags=[]
    layer=row.get("asap_layer",""); severity=get_severity(row.get("Votes",1))
    tactics,techniques,descriptions=set(),[],[]
    for tag in tags:
        tag=tag.strip()
        if tag in STRIDE_TO_MITRE:
            m=STRIDE_TO_MITRE[tag]; tactics.add(f"{m['tactic']} ({m['tactic_id']})")
            techniques.append(f"{m['technique']} ({m['technique_id']})"); descriptions.append(m["description"])
    if layer in LAYER_TO_MITRE:
        m=LAYER_TO_MITRE[layer]; tactics.add(f"{m['tactic']} ({m['tactic_id']})")
        techniques.append(f"{m['technique']} ({m['technique_id']})")

    nist_entries=[]
    for tag in tags: nist_entries+=STRIDE_TO_NIST.get(tag.strip(),[])
    nist_entries+=LAYER_TO_NIST.get(layer,[])

    # Filter to only the priority NIST function for this severity level
    priority_fn = SEVERITY_TO_PRIORITY_FUNCTION.get(severity, "Protect")
    priority_entries = [e for e in nist_entries if e["function"] == priority_fn]
    if not priority_entries:
        priority_entries = nist_entries  # fallback if no match

    return pd.Series({
        "STRIDE Tags":      ", ".join(tags) if tags else "Unknown",
        "ASAP Layer":       layer,
        "Severity":         severity,
        "MITRE Tactics":    " | ".join(tactics)    if tactics    else "Unknown",
        "MITRE Techniques": " | ".join(techniques) if techniques else "Unknown",
        "NIST CSF Function":" | ".join(sorted(set(e["function"] for e in priority_entries))) or "Unknown",
        "NIST Category":    " | ".join(sorted(set(e["category"]  for e in priority_entries))) or "Unknown",
        "NIST Controls":    " | ".join(sorted(set(e["controls"]  for e in priority_entries))) or "Unknown",
        "Description":      " ".join(descriptions) if descriptions else "No description available.",
    })

def colour_severity(val):
    if val=="High":   return "background-color:#5c1a1a;color:white"
    if val=="Medium": return "background-color:#5c3d1a;color:white"
    if val=="Low":    return "background-color:#1a3a1a;color:white"
    return ""

def colour_nist(val):
    for fn,col in NIST_FUNCTION_COLORS.items():
        if fn in str(val): return f"background-color:{col}22;color:white"
    return ""

def colour_status(val):
    return f"background-color:{STATUS_COLORS.get(val,'#333')};color:white"

def dark_ax(ax,fig):
    ax.set_facecolor("#0e1117"); fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    ax.spines[:].set_color("#444")

def build_pdf(catalogue, siem_df, summary_stats, raw_votes):
    from fpdf import FPDF
    from datetime import datetime

    def sanitise(text):
        return (str(text)
            .replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2022", "*")
            .replace("\u00a0", " ")
            .replace("\u2192", "->")
            .encode("latin-1", errors="replace").decode("latin-1")
        )

    class PDF(FPDF):
        def header(self):
            pass
        def footer(self):
            self.set_y(-12)
            self.set_font("helvetica","I",7)
            self.set_text_color(150,150,150)
            self.cell(0,8,sanitise(f"CBDC Threat Report  |  Page {self.page_no()}"),align="C")

    def write_table(pdf, df, title, col_widths=None, font_size=7):
        pdf.add_page()
        pdf.set_font("helvetica","B",13)
        pdf.set_text_color(30,30,80)
        pdf.cell(0,10,sanitise(title),ln=True)
        pdf.ln(2)
        cols = df.columns.tolist()
        page_w = pdf.w - pdf.l_margin - pdf.r_margin
        if col_widths is None:
            col_widths = [page_w / len(cols)] * len(cols)
        row_h = 5

        def draw_header():
            pdf.set_font("helvetica","B",font_size)
            pdf.set_fill_color(30,30,80); pdf.set_text_color(255,255,255)
            for c,w in zip(cols, col_widths):
                pdf.cell(w, 8, sanitise(str(c))[:30], border=1, fill=True)
            pdf.ln()

        draw_header()
        pdf.set_font("helvetica", size=font_size)
        for _, row in df.iterrows():
            sev = row.get("Severity","")
            status = row.get("Status","")
            if sev=="High" or status=="ESCALATED":
                pdf.set_fill_color(92,26,26); pdf.set_text_color(255,255,255)
            elif sev=="Medium" or status=="RECOVERING":
                pdf.set_fill_color(92,61,26); pdf.set_text_color(255,255,255)
            elif status=="AUTO_CONTAINED":
                pdf.set_fill_color(26,42,80); pdf.set_text_color(255,255,255)
            else:
                pdf.set_fill_color(26,58,26); pdf.set_text_color(255,255,255)

            max_lines = 1
            for c,w in zip(cols, col_widths):
                text = sanitise(str(row.get(c,"")))
                chars_per_line = max(1, int(w / (font_size * 0.45)))
                lines = max(1, -(-len(text) // chars_per_line))
                max_lines = max(max_lines, lines)
            cell_h = max(row_h, max_lines * (font_size * 0.52 + 1))

            if pdf.get_y() + cell_h > pdf.h - pdf.b_margin - 15:
                pdf.add_page()
                draw_header()
                pdf.set_font("helvetica", size=font_size)
                if sev=="High" or status=="ESCALATED":
                    pdf.set_fill_color(92,26,26); pdf.set_text_color(255,255,255)
                elif sev=="Medium" or status=="RECOVERING":
                    pdf.set_fill_color(92,61,26); pdf.set_text_color(255,255,255)
                elif status=="AUTO_CONTAINED":
                    pdf.set_fill_color(26,42,80); pdf.set_text_color(255,255,255)
                else:
                    pdf.set_fill_color(26,58,26); pdf.set_text_color(255,255,255)

            x_start = pdf.get_x()
            y_start = pdf.get_y()
            for c,w in zip(cols, col_widths):
                text = sanitise(str(row.get(c,"")))
                pdf.set_xy(x_start, y_start)
                pdf.multi_cell(w, row_h, text, border=1, fill=True)
                x_start += w
            pdf.set_xy(pdf.l_margin, y_start + cell_h)

    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(10); pdf.set_right_margin(10)

    # Cover
    pdf.add_page()
    pdf.set_fill_color(15,20,50); pdf.rect(0,0,297,210,"F")
    pdf.set_text_color(255,255,255); pdf.set_font("helvetica","B",22); pdf.ln(70)
    pdf.cell(0,14,sanitise("CBDC Automated Threat Detection"),ln=True,align="C")
    pdf.set_font("helvetica","B",14)
    pdf.cell(0,10,sanitise("Threat Catalogue | NIST CSF Controls | SIEM Automation"),ln=True,align="C")
    pdf.set_font("helvetica",size=11)
    pdf.cell(0,9,sanitise(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),ln=True,align="C")

    # Summary
    pdf.add_page()
    pdf.set_text_color(0,0,0); pdf.set_font("helvetica","B",14)
    pdf.cell(0,10,sanitise("Summary Statistics"),ln=True); pdf.set_font("helvetica",size=10); pdf.ln(2)
    for k,v in summary_stats.items():
        pdf.cell(0,8,sanitise(f"{k}: {v}"),ln=True)
    pdf.ln(4); pdf.set_font("helvetica","B",11); pdf.cell(0,8,sanitise("Severity Breakdown"),ln=True)
    pdf.set_font("helvetica",size=10)
    for sev,val in [("High",3),("Medium",2),("Low",1)]:
        pdf.cell(0,7,sanitise(f"  {sev}: {int((raw_votes==val).sum())}"),ln=True)
    if siem_df is not None and len(siem_df)>0:
        pdf.ln(4); pdf.set_font("helvetica","B",11); pdf.cell(0,8,sanitise("SIEM Incident Summary"),ln=True)
        pdf.set_font("helvetica",size=10)
        for s in ["ESCALATED","AUTO_CONTAINED","RECOVERING","MONITORING"]:
            count=len(siem_df[siem_df["Status"]==s]) if "Status" in siem_df.columns else 0
            pdf.cell(0,7,sanitise(f"  {s}: {count}"),ln=True)

    # Catalogue table
    page_w = 277
    cat_cols = catalogue.columns.tolist()
    cat_widths_map = {
        "Transaction #":   12,
        "STRIDE Tags":     30,
        "ASAP Layer":      18,
        "Severity":        14,
        "MITRE Tactics":   40,
        "MITRE Techniques":45,
        "NIST CSF Function":28,
        "NIST Category":   35,
        "NIST Controls":   28,
        "Description":     50,
    }
    cat_widths = [cat_widths_map.get(c, 25) for c in cat_cols]
    total = sum(cat_widths)
    cat_widths = [w * page_w / total for w in cat_widths]
    write_table(pdf, catalogue, "Threat Catalogue (MITRE ATT&CK + NIST CSF)", cat_widths, font_size=6)

    # SIEM table
    if siem_df is not None and len(siem_df)>0:
        siem_cols = siem_df.columns.tolist()
        siem_widths_map = {
            "Incident ID":     20,
            "Transaction #":   14,
            "Timestamp":       28,
            "Severity":        14,
            "Status":          22,
            "STRIDE Tags":     28,
            "ASAP Layer":      18,
            "MITRE Techniques":38,
            "NIST Controls":   28,
            "Triggered Rules": 45,
            "Auto Actions":    55,
            "Manual Actions":  35,
            "Escalate":        12,
        }
        siem_w = [siem_widths_map.get(c,20) for c in siem_cols]
        total_s = sum(siem_w)
        siem_w = [w * page_w / total_s for w in siem_w]
        write_table(pdf, siem_df, "SIEM Incident Log", siem_w, font_size=6)

    return bytes(pdf.output())


# ══════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════

st.title("CBDC Automated Threat Detection")
uploaded=st.file_uploader("Upload Transaction Log",type="csv")

if uploaded:
    df=pd.read_csv(uploaded)
    st.subheader("Uploaded Dataset"); st.dataframe(df)

    st.sidebar.header("Model Parameters")
    z      =st.sidebar.slider("Z-score Threshold",2.0,5.0,3.0)
    eps    =st.sidebar.slider("DBSCAN eps",0.1,2.0,0.8)
    samples=st.sidebar.slider("Minimum Samples",2,20,5)
    contam =st.sidebar.slider("Isolation Forest",0.01,0.20,0.05)

    if st.button("Start Threat Detection"):
        orig_process     = df["process"].values     if "process"     in df.columns else None
        orig_asap_layer  = df["asap_layer"].values  if "asap_layer"  in df.columns else None
        orig_stride_tags = df["stride_tags"].values if "stride_tags" in df.columns else None

        X,processed,feature_names,scaler = preprocess(df)
        zscore_result,dbscan_result,iso_result,votes,final = detect(X,z,eps,samples,contam)
        raw_votes = votes.copy()

        processed["ZScore"]          = zscore_result
        processed["DBSCAN"]          = dbscan_result
        processed["IsolationForest"] = iso_result
        processed["Votes"]           = raw_votes
        processed["Threat"]          = final

        if orig_process     is not None: processed["process"]     = orig_process
        if orig_asap_layer  is not None: processed["asap_layer"]  = orig_asap_layer
        if orig_stride_tags is not None: processed["stride_tags"] = orig_stride_tags

        st.session_state["detection_done"]   = True
        st.session_state["processed"]        = processed
        st.session_state["raw_votes"]        = raw_votes
        st.session_state["zscore_result"]    = zscore_result
        st.session_state["dbscan_result"]    = dbscan_result
        st.session_state["iso_result"]       = iso_result
        st.session_state["final"]            = final
        st.session_state["feature_names"]    = feature_names
        st.session_state["X"]                = X
        st.session_state["contam"]           = contam

    if st.session_state.get("detection_done", False):
        processed     = st.session_state["processed"]
        raw_votes     = st.session_state["raw_votes"]
        zscore_result = st.session_state["zscore_result"]
        dbscan_result = st.session_state["dbscan_result"]
        iso_result    = st.session_state["iso_result"]
        final         = st.session_state["final"]
        feature_names = st.session_state["feature_names"]
        X             = st.session_state["X"]
        contam        = st.session_state["contam"]

        st.success("Detection Completed")
        n_threat=int((raw_votes>=1).sum())
        c1,c2,c3=st.columns(3)
        c1.metric("Transactions",len(processed)); c2.metric("Threats",n_threat)
        c3.metric("Normal",len(processed)-n_threat)
        c4,c5,c6=st.columns(3)
        c4.metric("High",int((raw_votes==3).sum())); c5.metric("Medium",int((raw_votes==2).sum()))
        c6.metric("Low",int((raw_votes==1).sum()))

        threats_df=processed[raw_votes>=1].copy()

        # Summary stats
        st.subheader("Summary Statistics")
        risk_col=next((c for c in processed.columns if "risk" in c.lower()),None)
        avg_risk=f"{processed[risk_col].mean():.4f}" if risk_col else "N/A"
        all_tags_sm=[]
        if "stride_tags" in threats_df.columns:
            for t in threats_df["stride_tags"]:
                try: all_tags_sm+=ast.literal_eval(t)
                except: pass
        most_common=pd.Series(all_tags_sm).value_counts().idxmax() if all_tags_sm else "N/A"
        summary_stats={"Total Transactions":len(processed),"Total Threats":n_threat,
                       "High Severity":int((raw_votes==3).sum()),"Medium Severity":int((raw_votes==2).sum()),
                       "Low Severity":int((raw_votes==1).sum()),"Avg Risk Score":avg_risk,
                       "Most Common Threat":most_common}
        s1,s2,s3,s4=st.columns(4)
        s1.metric("Avg Risk Score",avg_risk); s2.metric("Most Common Threat",most_common)
        s3.metric("Detection Rate",f"{n_threat/len(processed)*100:.1f}%")
        s4.metric("High+Medium",int((raw_votes>=2).sum()))

        # Distributions
        st.subheader("Threat Distribution")
        dt1,dt2,dt3=st.tabs(["By Process","By ASAP Layer","By STRIDE Tag"])
        with dt1:
            if "process" in threats_df.columns:
                counts=threats_df["process"].value_counts()
                fig,ax=plt.subplots(figsize=(8,4))
                ax.bar(counts.index.astype(str),counts.values,color="#4C72B0")
                ax.set_xlabel("Process"); ax.set_ylabel("Count"); ax.set_title("Threats by Process")
                dark_ax(ax,fig); st.pyplot(fig)
        with dt2:
            if "asap_layer" in threats_df.columns:
                counts=threats_df["asap_layer"].value_counts()
                fig,ax=plt.subplots(figsize=(6,4))
                ax.bar(counts.index.astype(str),counts.values,color="#DD8452")
                ax.set_xlabel("ASAP Layer"); ax.set_ylabel("Count"); ax.set_title("Threats by ASAP Layer")
                dark_ax(ax,fig); st.pyplot(fig)
        with dt3:
            if "stride_tags" in threats_df.columns:
                all_tags=[]
                for t in threats_df["stride_tags"]:
                    try: all_tags+=ast.literal_eval(t)
                    except: pass
                if all_tags:
                    tag_counts=pd.Series(all_tags).value_counts()
                    fig,ax=plt.subplots(figsize=(8,4))
                    ax.barh(tag_counts.index.astype(str),tag_counts.values,color="#55A868")
                    ax.set_xlabel("Count"); ax.set_title("Threats by STRIDE Tag")
                    dark_ax(ax,fig); st.pyplot(fig)

        # Feature importance
        st.subheader("Feature Importance")
        X_df=pd.DataFrame(X,columns=feature_names)
        importance=(X_df[raw_votes>=1].mean()-X_df[raw_votes==0].mean()).abs().sort_values(ascending=True)
        top_n=importance.tail(15)
        fig,ax=plt.subplots(figsize=(8,5))
        ax.barh(top_n.index,top_n.values,color=cm.RdYlGn(np.linspace(0.2,0.9,len(top_n))))
        ax.set_xlabel("Mean Absolute Difference (Threat vs Normal)")
        ax.set_title("Top Features Contributing to Threats")
        dark_ax(ax,fig); st.pyplot(fig)

        # Agreement matrix
        st.subheader("Algorithm Agreement Matrix")
        agree_df=pd.DataFrame({"Z-Score":zscore_result,"DBSCAN":dbscan_result,"Isolation Forest":iso_result})
        keys=agree_df.columns.tolist()
        agree_matrix=pd.DataFrame(index=keys,columns=keys,dtype=float)
        for a in keys:
            for b in keys: agree_matrix.loc[a,b]=float((agree_df[a]==agree_df[b]).mean())
        fig,ax=plt.subplots(figsize=(5,4))
        im=ax.imshow(agree_matrix.values.astype(float),cmap="Blues",vmin=0,vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(keys,rotation=20,color="white",fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(keys,color="white",fontsize=9)
        for i in range(3):
            for j in range(3): ax.text(j,i,f"{agree_matrix.iloc[i,j]:.0%}",ha="center",va="center",color="black",fontsize=11)
        ax.set_title("Agreement Rate Between Algorithms",color="white")
        plt.colorbar(im,ax=ax); dark_ax(ax,fig); st.pyplot(fig)

        # Filters
        st.subheader("Threat Transactions")
        cf1,cf2,cf3=st.columns(3)

        proc_options  = sorted(threats_df["process"].astype(str).dropna().unique().tolist())    if "process"    in threats_df.columns else []
        layer_options = sorted(threats_df["asap_layer"].astype(str).dropna().unique().tolist()) if "asap_layer" in threats_df.columns else []

        f_proc   = cf1.multiselect("Filter by Process",    proc_options,  key="f_proc")
        f_layer  = cf2.multiselect("Filter by ASAP Layer", layer_options, key="f_layer")
        f_stride = cf3.multiselect("Filter by STRIDE Tag",
                                   ["Spoofing","Tampering","InformationDisclosure",
                                    "DenialOfService","Repudiation","ElevationOfPrivilege"],
                                   key="f_stride")
        search_id = st.text_input("Search by Transaction Index (row number)", key="search_id")

        filtered = threats_df.copy()
        if f_proc and "process" in filtered.columns:
            filtered = filtered[filtered["process"].astype(str).isin(f_proc)]
        if f_layer and "asap_layer" in filtered.columns:
            filtered = filtered[filtered["asap_layer"].astype(str).isin(f_layer)]
        if f_stride and "stride_tags" in filtered.columns:
            def has_tag(t):
                try: return any(s in ast.literal_eval(str(t)) for s in f_stride)
                except: return False
            filtered = filtered[filtered["stride_tags"].apply(has_tag)]
        if search_id.strip():
            try:
                idx = int(search_id.strip())
                if idx in filtered.index:
                    filtered = filtered.loc[[idx]]
                else:
                    st.warning(f"Transaction index {idx} not found in threats.")
            except ValueError:
                st.warning("Enter a valid integer row index.")

        st.dataframe(filtered, use_container_width=True)

        # Normal vs threat
        st.subheader("Normal vs Threat Transaction Comparison")
        normal_df=processed[raw_votes==0]
        if len(threats_df)>0 and len(normal_df)>0:
            st.dataframe(pd.DataFrame({"Normal":normal_df.iloc[0],"Threat":threats_df.iloc[0]}).T,use_container_width=True)

        # Why flagged
        st.subheader("Why Was It Flagged?")
        if len(threats_df)>0:
            X_df_t=X_df[raw_votes>=1]; explanations=[]
            for idx in X_df_t.index:
                row_z=np.abs(X_df.loc[idx]); top_feats=row_z.nlargest(3)
                reason=", ".join([f"{f} (z={v:.2f})" for f,v in top_feats.items()])
                explanations.append({"Transaction #":idx,"Top Contributing Features":reason,"Severity":get_severity(raw_votes[idx])})
            st.dataframe(pd.DataFrame(explanations),use_container_width=True)

        # SHAP
        st.subheader("SHAP - Isolation Forest Explainability")
        try:
            import shap
            iso_model=IsolationForest(contamination=contam,random_state=42); iso_model.fit(X)
            explainer=shap.TreeExplainer(iso_model); shap_values=explainer.shap_values(X)
            shap_df=pd.DataFrame(shap_values,columns=feature_names)
            mean_shap=shap_df.abs().mean().sort_values(ascending=False).head(15)
            fig,ax=plt.subplots(figsize=(8,5))
            ax.barh(mean_shap.index[::-1],mean_shap.values[::-1],color="#C44E52")
            ax.set_xlabel("Mean |SHAP value|"); ax.set_title("SHAP Feature Importance (Isolation Forest)")
            dark_ax(ax,fig); st.pyplot(fig)
            shap_threat=shap_df[raw_votes>=1]
            if len(shap_threat)>0:
                st.caption("SHAP values for first flagged transaction:")
                s_row=shap_threat.iloc[0].sort_values(key=abs,ascending=False).head(10)
                fig,ax=plt.subplots(figsize=(8,4))
                ax.barh(s_row.index[::-1],s_row.values[::-1],
                        color=["#C44E52" if v>0 else "#4C72B0" for v in s_row.values[::-1]])
                ax.axvline(0,color="white",lw=0.8)
                ax.set_xlabel("SHAP value"); ax.set_title("SHAP - First Flagged Transaction")
                dark_ax(ax,fig); st.pyplot(fig)
        except ImportError:
            st.warning("SHAP not installed. Run `.venv/bin/pip install shap` in terminal then restart.")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

        # Threat catalogue
        st.subheader("Threat Catalogue (MITRE ATT&CK + NIST CSF Controls)")
        st.markdown(
            "MITRE ATT&CK Techniques reference: "
            "[attack.mitre.org/techniques/enterprise]"
            "(https://attack.mitre.org/techniques/enterprise/)",
            unsafe_allow_html=False
        )
        siem_df=None
        if len(threats_df)==0:
            st.info("No threats detected.")
        else:
            catalogue=threats_df.apply(map_to_catalogue,axis=1)
            catalogue.insert(0,"Transaction #",threats_df.index)

            st.markdown("**NIST CSF Function Legend**")
            leg_cols=st.columns(5)
            for i,(fn,col) in enumerate(NIST_FUNCTION_COLORS.items()):
                leg_cols[i].markdown(f"<span style='background:{col};padding:3px 10px;border-radius:4px;color:white;font-size:12px;font-weight:bold'>{fn}</span>",unsafe_allow_html=True)
            st.markdown("")
            st.caption("NIST Controls are filtered to the most actionable function per severity: Low -> Detect, Medium -> Protect, High -> Respond.")
            styled=(catalogue.style.map(colour_severity,subset=["Severity"]).map(colour_nist,subset=["NIST CSF Function"]))
            st.dataframe(styled,use_container_width=True)

            st.markdown("**NIST CSF Function Distribution**")
            nist_funcs=[]
            for v in catalogue["NIST CSF Function"]: nist_funcs+=[f.strip() for f in str(v).split("|")]
            nist_counts=pd.Series(nist_funcs).value_counts().reindex(["Identify","Protect","Detect","Respond","Recover"],fill_value=0)
            fig,ax=plt.subplots(figsize=(7,3))
            ax.bar(nist_counts.index,nist_counts.values,color=[NIST_FUNCTION_COLORS.get(k,"#888") for k in nist_counts.index])
            ax.set_ylabel("Count"); ax.set_title("Threats by NIST CSF Function")
            dark_ax(ax,fig); st.pyplot(fig)

            # SIEM
            st.subheader("SIEM-Based Security Controls Automation")
            st.caption("Each flagged threat is automatically matched to SIEM rules. Actions are fired based on STRIDE tag + ASAP layer + severity.")

            if "siem_incidents" not in st.session_state or st.session_state.get("siem_catalogue_hash") != hash(str(catalogue.values.tobytes())):
                engine=SiemEngine(); incidents=engine.process_all(catalogue)
                siem_df=engine.to_dataframe(); siem_sum=engine.summary()
                st.session_state["siem_incidents"]      = incidents
                st.session_state["siem_df"]             = siem_df
                st.session_state["siem_sum"]            = siem_sum
                st.session_state["siem_catalogue_hash"] = hash(str(catalogue.values.tobytes()))
            else:
                incidents = st.session_state["siem_incidents"]
                siem_df   = st.session_state["siem_df"]
                siem_sum  = st.session_state["siem_sum"]

            sm1,sm2,sm3,sm4=st.columns(4)
            sm1.metric("Total Incidents",siem_sum.get("Total Incidents",0))
            sm2.metric("Auto Contained", siem_sum.get("Auto Contained",0))
            sm3.metric("Escalated",      siem_sum.get("Escalated",0))
            sm4.metric("Rules Fired",    siem_sum.get("Rules Fired",0))

            status_counts=siem_df["Status"].value_counts()
            fig,ax=plt.subplots(figsize=(7,3))
            ax.bar(status_counts.index,status_counts.values,
                   color=[STATUS_COLORS.get(s,"#555") for s in status_counts.index])
            ax.set_ylabel("Count"); ax.set_title("SIEM Incident Status Distribution")
            dark_ax(ax,fig); st.pyplot(fig)

            st.markdown("**SIEM Incident Log**")
            siem_styled=siem_df.style.map(colour_severity,subset=["Severity"]).map(colour_status,subset=["Status"])
            st.dataframe(siem_styled,use_container_width=True)

            st.markdown("**Incident Detail View**")
            selected_inc=st.selectbox("Select Incident to Inspect",siem_df["Incident ID"].tolist(),key="inc_select")
            if selected_inc:
                inc=next((i for i in incidents if i.incident_id==selected_inc),None)
                if inc:
                    d1,d2,d3=st.columns(3)
                    d1.metric("Severity",inc.severity); d2.metric("Status",inc.status)
                    d3.metric("Escalate","Yes" if inc.escalate else "No")
                    st.markdown(f"**STRIDE Tags:** {', '.join(inc.stride_tags)}")
                    st.markdown(f"**ASAP Layer:** {inc.asap_layer}")
                    st.markdown(f"**MITRE Techniques:** {inc.mitre_techniques}")
                    st.markdown(f"**NIST Controls:** {inc.nist_controls}")
                    st.markdown(f"**Triggered Rules:** {' | '.join(inc.triggered_rules)}")
                    st.markdown("**Automated Actions:**")
                    for a in [x for x in inc.actions if x.automated]:
                        color={"DETECT":"#DD8452","PROTECT":"#55A868","RESPOND":"#C44E52",
                               "RECOVER":"#8172B2","LOG":"#4C72B0","ESCALATE":"#cc0000"}.get(a.action_type,"#555")
                        st.markdown(f"<span style='background:{color};padding:2px 8px;border-radius:3px;color:white;font-size:11px'>{a.action_type}</span> &nbsp; `{a.nist_control}` - {a.description}",unsafe_allow_html=True)
                    manual=[x for x in inc.actions if not x.automated]
                    if manual:
                        st.markdown("**Manual Actions Required:**")
                        for a in manual: st.markdown(f"Warning: `{a.nist_control}` - {a.description}")

            dl1,dl2=st.columns(2)
            with dl1: st.download_button("Download Catalogue (CSV)",catalogue.to_csv(index=False),"Threat_Catalogue.csv","text/csv")
            with dl2:
                try:
                    pdf_bytes=build_pdf(catalogue,siem_df,summary_stats,raw_votes)
                    st.download_button("Download Full Report (PDF)",pdf_bytes,"Threat_Report.pdf","application/pdf")
                except Exception as e:
                    st.error(f"PDF error: {e}")
            if siem_df is not None:
                st.download_button("Download SIEM Incident Log (CSV)",siem_df.to_csv(index=False),"SIEM_Incidents.csv","text/csv")

        # Model evaluation
        st.subheader("Model Evaluation")
        y_true=final

        algorithms={"Z-Score":zscore_result,"DBSCAN":dbscan_result,"Isolation Forest":iso_result,"Ensemble":final}
        scores={"Z-Score":zscore_result.astype(float),"DBSCAN":dbscan_result.astype(float),
                "Isolation Forest":iso_result.astype(float),"Ensemble":raw_votes.astype(float)/3}
        colors_ev={"Z-Score":"#4C72B0","DBSCAN":"#DD8452","Isolation Forest":"#55A868","Ensemble":"#C44E52"}

        can_draw_curves = len(np.unique(y_true)) >= 2

        tab1,tab2,tab3=st.tabs(["ROC Curve","Precision-Recall Curve","Confusion Matrix"])

        with tab1:
            fig,ax=plt.subplots(figsize=(7,5))
            if not can_draw_curves:
                ax.text(0.5,0.5,
                        "ROC curve unavailable:\nAll transactions have the same label.\n"
                        "Adjust model parameters to get both\nnormal and threat predictions.",
                        ha="center",va="center",color="white",fontsize=11,transform=ax.transAxes)
            else:
                ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
                for name,score in scores.items():
                    fpr,tpr,_=roc_curve(y_true,score)
                    ax.plot(fpr,tpr,color=colors_ev[name],lw=2,label=f"{name} (AUC={auc(fpr,tpr):.2f})")
                ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                ax.legend(facecolor="#1a1a2e",labelcolor="white",loc="lower right")
            ax.set_title("ROC Curve")
            dark_ax(ax,fig); st.pyplot(fig)

        with tab2:
            fig,ax=plt.subplots(figsize=(7,5))
            if not can_draw_curves:
                ax.text(0.5,0.5,
                        "Precision-Recall curve unavailable:\nAll transactions have the same label.\n"
                        "Adjust model parameters to get both\nnormal and threat predictions.",
                        ha="center",va="center",color="white",fontsize=11,transform=ax.transAxes)
            else:
                for name,score in scores.items():
                    prec,rec,_=precision_recall_curve(y_true,score)
                    ax.plot(rec,prec,color=colors_ev[name],lw=2,label=f"{name} (AUC={auc(rec,prec):.2f})")
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
                ax.legend(facecolor="#1a1a2e",labelcolor="white",loc="upper right")
            ax.set_title("Precision-Recall Curve")
            dark_ax(ax,fig); st.pyplot(fig)

        with tab3:
            cms_cols=st.columns(2)
            for i,(name,preds) in enumerate(algorithms.items()):
                with cms_cols[i%2]:
                    fig,ax=plt.subplots(figsize=(4,3.5))
                    disp=ConfusionMatrixDisplay(confusion_matrix(y_true,preds),display_labels=["Normal","Threat"])
                    disp.plot(ax=ax,colorbar=False,cmap="Blues")
                    ax.set_title(name,color="white"); dark_ax(ax,fig)
                    for text in ax.texts: text.set_color("black")
                    st.pyplot(fig)