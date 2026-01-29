import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list

st.set_page_config(page_title="Compliance Dashboard", layout="wide")
st.title("Compliance Dashboard")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

# ======================================================
# Helpers
# ======================================================
def normalize_values(df):
    def norm_compliance(x):
        x = "" if pd.isna(x) else str(x).strip().lower()
        if x in ["yes", "y", "1", "true"]:
            return "Compliant"
        if x in ["no", "n", "0", "false"]:
            return "Not compliant"
        if "partial" in x:
            return "Partial"
        return "Blank"

    df["Compliance_status"] = df["Compliance_raw"].apply(norm_compliance)
    df["Goal_date"] = pd.to_datetime(df["Goal_raw"], format="%d.%m.%Y", errors="coerce")

    score_map = {"Compliant": 1.0, "Partial": 0.5}
    df["Score"] = df["Compliance_status"].map(score_map).fillna(0.0)

    df["Is_open"] = df["Compliance_status"].isin(["Partial", "Not compliant"])
    return df


def read_excel_two_header(uploaded_file, sheet):
    df = pd.read_excel(uploaded_file, sheet_name=sheet, header=[0, 1])
    tuples, last = [], None
    for d, f in df.columns:
        d = "" if pd.isna(d) else str(d).strip()
        f = "" if pd.isna(f) else str(f).strip()
        if d:
            last = d
        else:
            d = last
        tuples.append((d, f))
    df.columns = pd.MultiIndex.from_tuples(tuples)
    return df


def to_long(df):
    crit_col = next(c for c in df.columns if "criteria" in str(c).lower())
    criteria = df[crit_col]
    order = criteria.dropna().astype(str).str.strip().tolist()

    rest = df.drop(columns=[crit_col])
    stacked = rest.stack(level=0).reset_index().rename(columns={"level_1": "Department"})

    comp_col = next(c for c in stacked.columns if "compliant" in str(c).lower())
    goal_col = next(c for c in stacked.columns if "goal" in str(c).lower())

    out = pd.DataFrame({
        "Department": stacked["Department"],
        "Criteria": criteria.loc[stacked["level_0"]].values,
        "Compliance_raw": stacked[comp_col].values,
        "Goal_raw": stacked[goal_col].values
    })

    out = out.dropna(subset=["Criteria"])
    out["Criteria"] = out["Criteria"].astype(str).str.strip()
    return out, list(dict.fromkeys(order))


# ======================================================
# App
# ======================================================
if not uploaded:
    st.info("Upload an Excel file to begin.")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet = st.selectbox("Select sheet", xls.sheet_names)

df_wide = read_excel_two_header(uploaded, sheet)
df_long, criteria_order = to_long(df_wide)
df_long = normalize_values(df_long)

df_long["Criteria"] = pd.Categorical(df_long["Criteria"], categories=criteria_order, ordered=True)

today = pd.Timestamp.today().normalize()

# ======================================================
# Tabs
# ======================================================
tab_exec, tab_overview, tab_cluster = st.tabs(
    ["Executive Summary", "Overview Heatmap", "Clustered Heatmap"]
)

# ======================================================
# 1️⃣ EXECUTIVE SUMMARY TAB
# ======================================================
with tab_exec:
    st.subheader("Executive Compliance Summary")

    overall = df_long["Score"].mean() * 100
    open_items = df_long[df_long["Is_open"]]
    overdue = open_items[open_items["Goal_date"] < today]

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Compliance", f"{overall:.1f}%")
    c2.metric("Open Items", len(open_items))
    c3.metric("Overdue Items", len(overdue))

    weakest_criteria = (
        df_long.groupby("Criteria")["Score"].mean()
        .sort_values()
        .head(3)
        .index.tolist()
    )

    weakest_departments = (
        df_long.groupby("Department")["Score"].mean()
        .sort_values()
        .head(3)
        .index.tolist()
    )

    st.markdown("### Key Insights")
    st.markdown(
        f"""
- Overall compliance is **{overall:.1f}%**.
- There are **{len(open_items)} open items**, with **{len(overdue)} overdue**.
- Weakest criteria: **{", ".join(weakest_criteria)}**.
- Departments requiring attention: **{", ".join(weakest_departments)}**.
        """
    )

    st.markdown("### Recommended Actions")
    st.markdown(
        """
- Prioritize overdue items in low-performing departments.
- Address systemic issues in weak criteria.
- Align similar departments with shared remediation plans.
        """
    )

# ======================================================
# 2️⃣ IMPROVED HEATMAP TAB
# ======================================================
with tab_overview:
    st.subheader("Compliance Heatmap (Sorted & Enhanced)")

    pivot = df_long.pivot_table(
        index="Department",
        columns="Criteria",
        values="Score",
        aggfunc="max"
    ).fillna(0)

    # Sort departments by average compliance
    dept_avg = pivot.mean(axis=1).sort_values()
    pivot = pivot.loc[dept_avg.index]
    pivot = pivot.reindex(columns=criteria_order)

    # Add averages
    pivot["Dept Avg"] = pivot.mean(axis=1)
    crit_avg = pivot.mean(axis=0)
    pivot.loc["Criteria Avg"] = crit_avg

    fig = px.imshow(
        pivot,
        aspect="auto",
        title="Compliance Heatmap (Departments sorted by performance)"
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Green = compliant, amber = partial, grey = non-compliant.")

# ======================================================
# 3️⃣ CLUSTERED HEATMAP TAB
# ======================================================
with tab_cluster:
    st.subheader("Clustered Heatmap (Department Similarity)")

    base = df_long.pivot_table(
        index="Department",
        columns="Criteria",
        values="Score",
        aggfunc="mean"
    ).fillna(0)

    # Hierarchical clustering on departments
    linkage_matrix = linkage(base.values, method="ward")
    order = leaves_list(linkage_matrix)

    clustered = base.iloc[order]
    clustered = clustered.reindex(columns=criteria_order)

    fig_cluster = px.imshow(
        clustered,
        aspect="auto",
        title="Clustered Compliance Heatmap (Similar departments grouped)"
    )
    fig_cluster.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown(
        """
**How to read this:**
- Departments close together have similar compliance patterns.
- Use this to design **shared remediation actions**.
        """
    )
