import re
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional: clustering / extra stats could use SciPy later, but not needed for this simple version.

st.set_page_config(page_title="Compliance Dashboard (Simple)", layout="wide")
st.title("Compliance Dashboard (Simple)")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])


# ======================================================
# Data helpers
# ======================================================
def normalize_values(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()

    def norm_compliance(x):
        x = "" if pd.isna(x) else str(x).strip()
        xl = x.lower()
        if xl in ["yes", "y", "1", "true"]:
            return "Compliant"
        if xl in ["no", "n", "0", "false"]:
            return "Not compliant"
        if "partial" in xl:
            return "Partial"
        if x == "":
            return "Blank"
        return x

    def norm_goal(x):
        x = "" if pd.isna(x) else str(x).strip()
        if x.upper() == "MISSING":
            return "Missing"
        if x == "":
            return "Blank"
        return x

    df["Compliance_status"] = df["Compliance_raw"].apply(norm_compliance)
    df["Goal_status"] = df["Goal_raw"].apply(norm_goal)

    # dd.mm.yyyy
    df["Goal_date"] = pd.to_datetime(df["Goal_raw"], format="%d.%m.%Y", errors="coerce")

    score_map = {"Compliant": 1.0, "Partial": 0.5}
    df["Score"] = df["Compliance_status"].map(score_map).fillna(0.0)

    df["Is_open_item"] = df["Compliance_status"].isin(["Partial", "Not compliant"])
    df["Has_goal_date"] = df["Goal_date"].notna()
    df["Has_missing_goal"] = df["Goal_status"].eq("Missing")

    return df


def read_excel_two_header(uploaded_file, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=[0, 1])

    # Forward-fill dept names (merged cells -> blanks)
    tuples = []
    last_dept = None
    for dept, field in df.columns:
        dept = "" if pd.isna(dept) else str(dept).strip()
        field = "" if pd.isna(field) else str(field).strip()

        if dept:
            last_dept = dept
        else:
            dept = last_dept

        tuples.append((dept, field))

    df.columns = pd.MultiIndex.from_tuples(tuples)
    return df


def to_long_with_order(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Find criteria column
    criteria_col = None
    for col in df.columns:
        if "criteria" in str(col[0]).lower() or "criteria" in str(col[1]).lower():
            criteria_col = col
            break
    if criteria_col is None:
        raise ValueError("Couldn't find the Criteria column. Make sure the header contains 'Criteria'.")

    criteria_series = df[criteria_col].rename("Criteria")
    criteria_order = (
        criteria_series.dropna()
        .astype(str).str.strip()
        .loc[lambda s: (s != "") & (s.str.lower() != "nan")]
        .drop_duplicates()
        .tolist()
    )

    rest = df.drop(columns=[criteria_col])
    stacked = rest.stack(level=0).reset_index().rename(columns={"level_1": "Department"})

    lower_map = {c: str(c).lower() for c in stacked.columns}
    compliance_col = next((c for c, v in lower_map.items() if "compliant" in v), None)
    goal_col = next((c for c, v in lower_map.items() if "goal" in v), None)
    if compliance_col is None or goal_col is None:
        raise ValueError("Couldn't detect 'Compliant' and 'Goal' columns under each department.")

    out = pd.DataFrame({
        "Department": stacked["Department"],
        "Criteria": criteria_series.loc[stacked["level_0"]].values,
        "Compliance_raw": stacked[compliance_col].values,
        "Goal_raw": stacked[goal_col].values,
    })

    out["Criteria"] = out["Criteria"].astype(str).str.strip()
    out = out[out["Criteria"].ne("") & (out["Criteria"].str.lower() != "nan")]
    return out, criteria_order


def heatmap_with_zoom(
    data: pd.DataFrame,
    title: str,
    key_prefix: str,
    criteria_order: list[str] | None = None,
):
    if data.empty:
        st.info("No data to display.")
        return

    # preserve criteria order if given
    if criteria_order is not None:
        cols = [c for c in criteria_order if c in data.columns]
        rest = [c for c in data.columns if c not in cols]
        data = data[cols + rest]

    zoom = st.slider("Heatmap zoom (px per column)", 20, 80, 40, key=f"{key_prefix}_zoom")

    n_rows, n_cols = data.shape
    fig = px.imshow(data, aspect="auto", title=title)
    fig.update_layout(
        width=max(900, zoom * n_cols),
        height=max(500, 22 * n_rows),
        autosize=False,
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_showscale=False
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=11))

    st.plotly_chart(fig, use_container_width=False)


# ======================================================
# App start
# ======================================================
if not uploaded:
    st.info("Upload an Excel file to begin.")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet = st.selectbox("Select sheet", xls.sheet_names)

df_wide = read_excel_two_header(uploaded, sheet)

with st.expander("Preview (as uploaded)"):
    st.dataframe(df_wide.head(15), use_container_width=True)

try:
    df_long, criteria_order = to_long_with_order(df_wide)
except Exception as e:
    st.error(f"Failed to reshape your Excel file: {e}")
    st.stop()

df_long = normalize_values(df_long)
df_long["Criteria"] = pd.Categorical(df_long["Criteria"], categories=criteria_order, ordered=True)

today = pd.Timestamp.now().normalize()

departments = sorted(df_long["Department"].dropna().unique())

# ======================================================
# Tabs (Simple)
# ======================================================
tab1, tab2, tab3 = st.tabs(
    ["1) Overview (All departments)", "2) Department view", "3) Problem areas"]
)

# ======================================================
# 1) Overview (All departments)
# ======================================================
with tab1:
    st.subheader("Overview (All Departments)")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    overall = df_long["Score"].mean() * 100 if len(df_long) else 0
    open_items = df_long[df_long["Is_open_item"]]
    overdue = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] < today)]
    missing_goals = df_long[df_long["Has_missing_goal"]]

    c1.metric("Overall compliance", f"{overall:.1f}%")
    c2.metric("Open items", int(open_items.shape[0]))
    c3.metric("Overdue goals", int(overdue.shape[0]))
    c4.metric("Missing goals", int(missing_goals.shape[0]))

    # Heatmap: Department x Criteria
    pivot = df_long.pivot_table(index="Department", columns="Criteria", values="Score", aggfunc="max").fillna(0)
    pivot = pivot.reindex(columns=criteria_order)

    # Sort departments by avg compliance (worst first) for decision making
    dept_avg = pivot.mean(axis=1).sort_values(ascending=True)
    pivot = pivot.loc[dept_avg.index]

    # Add averages
    pivot2 = pivot.copy()
    pivot2["Dept Avg"] = pivot2.mean(axis=1)
    crit_avg = pivot.mean(axis=0)
    crit_avg["Dept Avg"] = pivot2["Dept Avg"].mean()
    pivot2.loc["Criteria Avg"] = crit_avg

    heatmap_with_zoom(
        pivot2,
        title="Heatmap (1=Compliant, 0.5=Partial, 0=Not compliant) — sorted by lowest compliance",
        key_prefix="all_heat",
        criteria_order=criteria_order
    )

# ======================================================
# 2) Department view (Individual department)
# ======================================================
with tab2:
    st.subheader("Department view")

    dept = st.selectbox("Select department", departments, key="dept_select")
    d_dept = df_long[df_long["Department"] == dept].copy().sort_values("Criteria")

    # KPI row for department
    c1, c2, c3, c4 = st.columns(4)
    dept_compliance = d_dept["Score"].mean() * 100 if len(d_dept) else 0
    dept_open = d_dept[d_dept["Is_open_item"]]
    dept_overdue = dept_open[dept_open["Has_goal_date"] & (dept_open["Goal_date"] < today)]
    dept_missing_goals = d_dept[d_dept["Has_missing_goal"]]

    c1.metric("Compliance", f"{dept_compliance:.1f}%")
    c2.metric("Open items", int(dept_open.shape[0]))
    c3.metric("Overdue goals", int(dept_overdue.shape[0]))
    c4.metric("Missing goals", int(dept_missing_goals.shape[0]))

    # --- Department heatmap (single row is not useful); use a criteria score bar instead
    st.markdown("### Criteria heat view")
    crit_scores = d_dept[["Criteria", "Score", "Compliance_status"]].copy()
    fig_bar = px.bar(
        crit_scores,
        x="Score",
        y="Criteria",
        color="Compliance_status",
        orientation="h",
        title="Criteria scores (0=Not compliant, 0.5=Partial, 1=Compliant)"
    )
    fig_bar.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Radar
    st.markdown("### Radar")
    fig_radar = px.line_polar(
        d_dept,
        r="Score",
        theta="Criteria",
        line_close=True,
        range_r=[0, 1],
        title=f"{dept} – Compliance by Criteria"
    )
    fig_radar.update_traces(fill="toself")
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- Timeline (only open items with goal dates)
    st.markdown("### Timeline (open items with dates)")
    timeline_df = d_dept[d_dept["Is_open_item"] & d_dept["Has_goal_date"]].copy()
    if timeline_df.empty:
        st.info("No dated goals for open items in this department.")
    else:
        timeline_df["Days_from_now"] = (timeline_df["Goal_date"] - today).dt.days
        timeline_df = timeline_df.sort_values(["Goal_date", "Criteria"])

        fig_timeline = px.scatter(
            timeline_df,
            x="Goal_date",
            y="Criteria",
            color="Compliance_status",
            title=f"{dept} – Goal dates (open items)",
            labels={"Goal_date": "Target date"}
        )
        fig_timeline.update_traces(marker=dict(size=12))
        fig_timeline.update_layout(yaxis=dict(categoryorder="array", categoryarray=criteria_order))
        st.plotly_chart(fig_timeline, use_container_width=True)

        fig_days = px.bar(
            timeline_df,
            x="Days_from_now",
            y="Criteria",
            orientation="h",
            color="Compliance_status",
            title=f"{dept} – Days remaining (negative = overdue)",
            labels={"Days_from_now": "Days from now"}
        )
        fig_days.update_layout(yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=criteria_order))
        fig_days.add_vline(x=0, line_width=2, line_dash="dash")
        st.plotly_chart(fig_days, use_container_width=True)

    with st.expander("Raw rows (department)"):
        st.dataframe(
            d_dept[["Criteria", "Compliance_raw", "Goal_raw", "Compliance_status", "Goal_status", "Goal_date", "Score"]],
            use_container_width=True
        )

# ======================================================
# 3) Problem areas (most problematic criteria + departments)
# ======================================================
with tab3:
    st.subheader("Problem areas")

    # Criteria ranking (lowest average score)
    st.markdown("### Most problematic criteria (across all departments)")

    crit_stats = (
        df_long.groupby("Criteria")
        .agg(
            Avg_score=("Score", "mean"),
            Not_compliant=("Compliance_status", lambda s: int((s == "Not compliant").sum())),
            Partial=("Compliance_status", lambda s: int((s == "Partial").sum())),
            Compliant=("Compliance_status", lambda s: int((s == "Compliant").sum())),
            Total=("Compliance_status", "size"),
        )
        .reset_index()
    )
    crit_stats["Compliance_pct"] = (crit_stats["Avg_score"] * 100).round(1)
    crit_stats["Problem_count"] = crit_stats["Not_compliant"] + crit_stats["Partial"]
    crit_stats = crit_stats.sort_values(["Avg_score", "Problem_count"], ascending=[True, False])

    top_n = st.slider("How many criteria to show", 5, min(30, len(crit_stats)), 10, key="topn_crit")
    st.dataframe(crit_stats.head(top_n), use_container_width=True)

    fig_crit = px.bar(
        crit_stats.head(top_n),
        x="Compliance_pct",
        y="Criteria",
        orientation="h",
        title="Top problematic criteria (lowest compliance %)",
        labels={"Compliance_pct": "Avg compliance (%)"}
    )
    fig_crit.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_crit, use_container_width=True)

    # Departments ranking (lowest average score)
    st.markdown("### Departments needing attention")

    dept_stats = (
        df_long.groupby("Department")
        .agg(
            Avg_score=("Score", "mean"),
            Not_compliant=("Compliance_status", lambda s: int((s == "Not compliant").sum())),
            Partial=("Compliance_status", lambda s: int((s == "Partial").sum())),
            Missing_goals=("Has_missing_goal", "sum"),
            Overdue_goals=("Goal_date", lambda d: int(((pd.to_datetime(d, errors="coerce") < today) & d.notna()).sum())),
            Total=("Score", "size"),
        )
        .reset_index()
    )
    dept_stats["Compliance_pct"] = (dept_stats["Avg_score"] * 100).round(1)
    dept_stats["Problem_count"] = dept_stats["Not_compliant"] + dept_stats["Partial"]
    dept_stats = dept_stats.sort_values(["Compliance_pct", "Problem_count"], ascending=[True, False])

    st.dataframe(dept_stats, use_container_width=True)

    fig_dept = px.bar(
        dept_stats,
        x="Compliance_pct",
        y="Department",
        orientation="h",
        title="Departments ranked by avg compliance (%)",
        labels={"Compliance_pct": "Avg compliance (%)"}
    )
    fig_dept.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_dept, use_container_width=True)

    # Matrix: show where problems concentrate (only Partial/Not compliant)
    st.markdown("### Where are the problems concentrated? (Only Partial / Not compliant)")
    prob = df_long[df_long["Compliance_status"].isin(["Partial", "Not compliant"])].copy()
    if prob.empty:
        st.info("No Partial/Not compliant items found.")
    else:
        prob_pivot = prob.pivot_table(
            index="Criteria",
            columns="Department",
            values="Score",
            aggfunc="count"
        ).fillna(0).astype(int)

        # Order criteria and departments by totals
        crit_order = prob_pivot.sum(axis=1).sort_values(ascending=False).index.tolist()
        dept_order = prob_pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
        prob_pivot = prob_pivot.loc[crit_order, dept_order]

        heatmap_with_zoom(
            prob_pivot,
            title="Problem count heatmap (count of Partial/Not compliant) — Criteria × Department",
            key_prefix="prob_heat"
        )
