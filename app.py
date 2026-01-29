import pandas as pd
import streamlit as st
import plotly.express as px

# SciPy for clustering (optional but recommended)
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Compliance Dashboard", layout="wide")
st.title("Compliance Dashboard")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])


# ======================================================
# Data prep helpers
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

    df["Goal_date"] = pd.to_datetime(df["Goal_raw"], format="%d.%m.%Y", errors="coerce")

    score_map = {"Compliant": 1.0, "Partial": 0.5}
    df["Score"] = df["Compliance_status"].map(score_map).fillna(0.0)

    df["Is_open_item"] = df["Compliance_status"].isin(["Partial", "Not compliant"])
    df["Has_goal_date"] = df["Goal_date"].notna()
    df["Has_missing_goal"] = df["Goal_status"].eq("Missing")

    return df


def read_excel_two_header(uploaded_file, sheet_name: str) -> pd.DataFrame:
    """
    Reads Excel where:
    - Header row 1 has Department labels repeated: A A B B C C ...
    - Header row 2 has field names repeated: Compliant? Goal Compliant? Goal ...
    """
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=[0, 1])

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
        raise ValueError(
            "Couldn't detect 'Compliant' and 'Goal' columns in the second header row. "
            "Ensure the second header row contains words like 'Compliant?' and 'Goal'."
        )

    out = pd.DataFrame({
        "Department": stacked["Department"],
        "Criteria": criteria_series.loc[stacked["level_0"]].values,
        "Compliance_raw": stacked[compliance_col].values,
        "Goal_raw": stacked[goal_col].values,
    })

    out["Criteria"] = out["Criteria"].astype(str).str.strip()
    out = out[out["Criteria"].ne("") & (out["Criteria"].str.lower() != "nan")]

    return out, criteria_order


def dept_leaderboard(df_in: pd.DataFrame, today: pd.Timestamp, d30: pd.Timestamp) -> pd.DataFrame:
    grp = df_in.groupby("Department", dropna=True)

    out = grp.agg(
        Criteria_count=("Criteria", "nunique"),
        Compliance_pct=("Score", lambda s: round(float(s.mean() * 100), 1) if len(s) else 0.0),
        Open_items=("Is_open_item", "sum"),
        Missing_goals=("Has_missing_goal", "sum"),
        Dated_goals=("Has_goal_date", "sum"),
    ).reset_index()

    mask_open = df_in["Is_open_item"]
    mask_overdue = mask_open & df_in["Has_goal_date"] & (df_in["Goal_date"] < today)
    mask_due30 = mask_open & df_in["Has_goal_date"] & (df_in["Goal_date"] >= today) & (df_in["Goal_date"] <= d30)

    def count_mask_for_dept(dept: str, mask: pd.Series) -> int:
        idx = df_in["Department"] == dept
        return int(mask[idx].sum())

    out["Overdue"] = out["Department"].apply(lambda d: count_mask_for_dept(d, mask_overdue))
    out["Due_30d"] = out["Department"].apply(lambda d: count_mask_for_dept(d, mask_due30))

    def avg_days_open(dept: str):
        sub = df_in[(df_in["Department"] == dept) & mask_open & df_in["Has_goal_date"]]
        if sub.empty:
            return None
        return float((sub["Goal_date"] - today).dt.days.mean())

    out["Avg_days_to_goal_open"] = out["Department"].apply(avg_days_open).round(1)

    return out.sort_values(["Overdue", "Compliance_pct"], ascending=[False, True])


# ======================================================
# Visualization helper (FIXED widget keys)
# ======================================================
def draw_heatmap_with_controls(
    data: pd.DataFrame,
    title: str,
    criteria_order: list[str],
    key_prefix: str,
    show_chunking: bool = True,
    default_chunk: int = 12,
):
    """
    Heatmap with:
    - zoom slider (px per column)
    - optional chunking across criteria
    - bigger height, rotated labels
    - ALL widgets have unique keys via key_prefix (fixes duplicate ID errors)
    """
    if data.empty:
        st.info("No data to display (check filters).")
        return

    # enforce criteria order for columns that exist
    crit_cols = [c for c in criteria_order if c in data.columns]
    other_cols = [c for c in data.columns if c not in crit_cols]
    data = data[crit_cols + other_cols]

    # Optional chunking (only across criteria columns, not extra columns)
    view = data
    if show_chunking and len(crit_cols) > default_chunk:
        chunk = st.select_slider(
            "Criteria chunk size",
            options=[9, 12, 15, 18, 24],
            value=default_chunk,
            key=f"{key_prefix}_chunk"
        )
        start = st.number_input(
            "Start criterion (0-based)",
            min_value=0,
            max_value=max(0, len(crit_cols) - 1),
            value=0,
            step=int(chunk),
            key=f"{key_prefix}_start"
        )
        start = int(start)
        cols_view = crit_cols[start:start + int(chunk)]
        view = data[cols_view + other_cols]

    n_rows = view.shape[0]
    n_cols = view.shape[1]

    zoom = st.slider(
        "Heatmap zoom (px per column)",
        20, 80, 40,
        key=f"{key_prefix}_zoom",
        help="Higher = wider cells (horizontal scrolling enabled)."
    )

    fig = px.imshow(view, aspect="auto", title=title)
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
st.subheader("Preview (as uploaded)")
st.dataframe(df_wide.head(15), use_container_width=True)

try:
    df_long, criteria_order = to_long_with_order(df_wide)
except Exception as e:
    st.error(f"Failed to reshape your Excel file: {e}")
    st.stop()

df_long = normalize_values(df_long)
df_long["Criteria"] = pd.Categorical(df_long["Criteria"], categories=criteria_order, ordered=True)

today = pd.Timestamp.now().normalize()
d30 = today + pd.Timedelta(days=30)
d60 = today + pd.Timedelta(days=60)
d90 = today + pd.Timedelta(days=90)

# Sidebar filters
st.sidebar.header("Filters")
all_depts = sorted(df_long["Department"].dropna().unique())
dept_sel = st.sidebar.multiselect("Department", all_depts, default=all_depts)

status_options = ["Compliant", "Partial", "Not compliant", "Blank"]
status_sel = st.sidebar.multiselect(
    "Compliance status",
    status_options,
    default=["Compliant", "Partial", "Not compliant"]
)

dff = df_long[df_long["Department"].isin(dept_sel)]
dff = dff[dff["Compliance_status"].isin(status_sel)]

# Tabs
tab_exec, tab_overview, tab_timelines, tab_radar, tab_cross, tab_cluster = st.tabs(
    ["Executive Summary", "Overview", "Timelines", "Radar", "Cross-department Goals", "Clustered Heatmap"]
)

# ======================================================
# Executive Summary
# ======================================================
with tab_exec:
    st.subheader("Executive Summary (one screen)")

    overall = (dff["Score"].mean() * 100) if len(dff) else 0
    open_items = dff[dff["Is_open_item"]]
    overdue = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] < today)]
    due_30 = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d30)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall compliance", f"{overall:.1f}%")
    c2.metric("Open items", int(open_items.shape[0]))
    c3.metric("Overdue goals", int(overdue.shape[0]))
    c4.metric("Due in 30 days", int(due_30.shape[0]))

    crit_means = dff.groupby("Criteria")["Score"].mean().sort_values()
    dept_means = dff.groupby("Department")["Score"].mean().sort_values()

    weakest_criteria = [str(x) for x in crit_means.head(5).index.tolist()] if not crit_means.empty else []
    weakest_depts = [str(x) for x in dept_means.head(5).index.tolist()] if not dept_means.empty else []

    st.markdown("### Key insights")
    st.markdown(
        f"""
- **As of {today.date()}**, overall compliance is **{overall:.1f}%** across the selected departments.
- There are **{int(open_items.shape[0])} open items**, with **{int(overdue.shape[0])} overdue**.
- **{int(due_30.shape[0])} items are due within 30 days**.
- Weakest criteria: **{", ".join(weakest_criteria[:3]) if weakest_criteria else "—"}**
- Departments needing attention: **{", ".join(weakest_depts[:3]) if weakest_depts else "—"}**
        """
    )

    st.markdown("### Top urgent items (overdue first)")
    urgent = open_items[open_items["Has_goal_date"]].copy()
    urgent["Days_from_now"] = (urgent["Goal_date"] - today).dt.days
    urgent = urgent.sort_values(["Days_from_now", "Department", "Criteria"]).head(10)

    if urgent.empty:
        st.info("No dated goals for open items in the current selection.")
    else:
        st.dataframe(
            urgent[["Department", "Criteria", "Compliance_status", "Goal_date", "Days_from_now"]],
            use_container_width=True
        )

# ======================================================
# Overview (heatmap + leaderboard + density + action lists)
# ======================================================
with tab_overview:
    st.subheader("Overview")

    open_items = dff[dff["Is_open_item"]]
    overdue = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] < today)]
    due_30 = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d30)]

    c1, c2, c3, c4, c5 = st.columns(5)
    overall = (dff["Score"].mean() * 100) if len(dff) else 0
    c1.metric("Overall compliance", f"{overall:.1f}%")
    c2.metric("Open items", int(open_items.shape[0]))
    c3.metric("Overdue goals", int(overdue.shape[0]))
    c4.metric("Due in 30 days", int(due_30.shape[0]))
    c5.metric("Missing goals", int(dff[dff["Has_missing_goal"]].shape[0]))

    st.subheader("Compliance heatmap (wide + zoom + chunking)")

    pivot = dff.pivot_table(index="Department", columns="Criteria", values="Score", aggfunc="max").fillna(0)
    pivot = pivot.reindex(columns=criteria_order)

    if not pivot.empty:
        sort_mode = st.radio(
            "Sort departments by",
            ["Lowest compliance first", "Highest compliance first", "Alphabetical"],
            horizontal=True,
            key="overview_sort"
        )
        dept_avg = pivot.mean(axis=1)
        if sort_mode == "Lowest compliance first":
            pivot = pivot.loc[dept_avg.sort_values(ascending=True).index]
        elif sort_mode == "Highest compliance first":
            pivot = pivot.loc[dept_avg.sort_values(ascending=False).index]
        else:
            pivot = pivot.sort_index()

        pivot2 = pivot.copy()
        pivot2["Dept Avg"] = pivot2.mean(axis=1)

        crit_avg = pivot2.drop(columns=["Dept Avg"]).mean(axis=0)
        crit_avg["Dept Avg"] = pivot2["Dept Avg"].mean()
        pivot2.loc["Criteria Avg"] = crit_avg

        draw_heatmap_with_controls(
            pivot2,
            title="Heatmap: 1=Compliant, 0.5=Partial, 0=Not compliant (includes Dept Avg + Criteria Avg)",
            criteria_order=criteria_order,
            key_prefix="overview_heat"
        )
    else:
        st.info("No data to display (check filters).")

    st.subheader("Department leaderboard")
    leaderboard = dept_leaderboard(dff, today=today, d30=d30)
    st.dataframe(leaderboard, use_container_width=True)

    st.subheader("Goal density over time (open items only)")
    goal_density = open_items[open_items["Has_goal_date"]].copy()
    if goal_density.empty:
        st.info("No dated goals for open items in the current selection.")
    else:
        goal_density["Month"] = goal_density["Goal_date"].dt.to_period("M").dt.to_timestamp()
        density = (
            goal_density.groupby(["Month", "Department"], as_index=False)
            .size()
            .rename(columns={"size": "Goal_count"})
            .sort_values("Month")
        )
        fig_density = px.bar(
            density,
            x="Month",
            y="Goal_count",
            color="Department",
            barmode="stack",
            title="Open-item goals due by month (stacked by department)",
            labels={"Goal_count": "Goals due"}
        )
        st.plotly_chart(fig_density, use_container_width=True)

    st.subheader("Action lists")
    t_missing, t_overdue, t_due = st.tabs(["Missing goals", "Overdue", "Due soon (30/60/90)"])

    with t_missing:
        miss_tbl = dff[dff["Has_missing_goal"]][["Department", "Criteria", "Compliance_status", "Goal_raw"]]
        st.dataframe(miss_tbl, use_container_width=True)

    with t_overdue:
        overdue_tbl = overdue[["Department", "Criteria", "Compliance_status", "Goal_date"]].sort_values(
            ["Department", "Goal_date", "Criteria"]
        )
        st.dataframe(overdue_tbl, use_container_width=True)

    with t_due:
        window = st.radio("Choose window", ["30 days", "60 days", "90 days"], horizontal=True, key="due_window")
        if window == "30 days":
            due_tbl = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d30)]
            label_end = d30
        elif window == "60 days":
            due_tbl = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d60)]
            label_end = d60
        else:
            due_tbl = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d90)]
            label_end = d90

        st.caption(f"Open items with goal dates between {today.date()} and {label_end.date()}.")
        due_tbl = due_tbl.copy()
        due_tbl["Days_from_now"] = (due_tbl["Goal_date"] - today).dt.days
        st.dataframe(
            due_tbl[["Department", "Criteria", "Compliance_status", "Goal_date", "Days_from_now"]]
            .sort_values(["Goal_date", "Department", "Criteria"]),
            use_container_width=True
        )

# ======================================================
# Timelines (single dept)
# ======================================================
with tab_timelines:
    st.subheader("Goals timeline (single department)")

    dept_timeline = st.selectbox(
        "Select department for timeline",
        sorted(dff["Department"].dropna().unique()) if not dff.empty else all_depts,
        key="timeline_dept"
    )

    timeline_df = df_long[
        (df_long["Department"] == dept_timeline) &
        (df_long["Has_goal_date"]) &
        (df_long["Is_open_item"])
    ].copy()

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
            title=f"{dept_timeline} – Goal Dates (open items only)",
            labels={"Goal_date": "Target date"}
        )
        fig_timeline.update_traces(marker=dict(size=12))
        fig_timeline.update_layout(yaxis=dict(categoryorder="array", categoryarray=criteria_order))
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.subheader("Days from now to each goal date (negative = overdue)")
        fig_days = px.bar(
            timeline_df,
            x="Days_from_now",
            y="Criteria",
            orientation="h",
            color="Compliance_status",
            title=f"{dept_timeline} – Days Remaining (open items only)",
            labels={"Days_from_now": "Days from now"}
        )
        fig_days.update_layout(yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=criteria_order))
        fig_days.add_vline(x=0, line_width=2, line_dash="dash")
        st.plotly_chart(fig_days, use_container_width=True)

# ======================================================
# Radar
# ======================================================
with tab_radar:
    st.subheader("Criteria compliance radar")

    dept_radar = st.selectbox(
        "Select department for radar",
        sorted(dff["Department"].dropna().unique()) if not dff.empty else all_depts,
        key="radar_dept"
    )

    radar_df = df_long[df_long["Department"] == dept_radar].copy()
    if radar_df.empty:
        st.info("No data for this department.")
    else:
        radar_df = radar_df.sort_values("Criteria")
        fig_radar = px.line_polar(
            radar_df,
            r="Score",
            theta="Criteria",
            line_close=True,
            range_r=[0, 1],
            title=f"{dept_radar} – Compliance by Criteria"
        )
        fig_radar.update_traces(fill="toself")
        st.plotly_chart(fig_radar, use_container_width=True)

# ======================================================
# Cross-department goals
# ======================================================
with tab_cross:
    st.subheader("Cross-department goals timeline (select departments)")

    dept_plot = st.multiselect(
        "Choose departments to plot",
        options=all_depts,
        default=dept_sel,
        key="cross_depts"
    )
    only_open = st.checkbox("Show only open items (Partial / Not compliant)", value=True, key="cross_only_open")

    cross = df_long[df_long["Department"].isin(dept_plot)].copy()
    cross = cross[cross["Has_goal_date"]]
    if only_open:
        cross = cross[cross["Is_open_item"]]

    if cross.empty:
        st.info("No dated goals match your selection.")
    else:
        cross = cross.sort_values(["Criteria", "Goal_date"])
        fig_cross = px.scatter(
            cross,
            x="Goal_date",
            y="Criteria",
            color="Department",
            symbol="Compliance_status",
            title="Goals Timeline: X = Goal Date, Y = Criteria, Color = Department",
            labels={"Goal_date": "Goal date"}
        )
        fig_cross.update_traces(marker=dict(size=10))
        fig_cross.update_layout(yaxis=dict(categoryorder="array", categoryarray=criteria_order))
        st.plotly_chart(fig_cross, use_container_width=True)

# ======================================================
# Clustered heatmap (FIXED duplicate widget IDs)
# ======================================================
with tab_cluster:
    st.subheader("Clustered heatmap (departments grouped by similarity)")

    if not SCIPY_OK:
        st.error("SciPy is not installed. Add `scipy` to requirements.txt to enable clustering.")
    else:
        base = dff.pivot_table(index="Department", columns="Criteria", values="Score", aggfunc="mean").fillna(0)
        base = base.reindex(columns=criteria_order)

        if base.shape[0] < 2:
            st.info("Need at least 2 departments in the filter selection to compute clusters.")
        else:
            Z = linkage(base.values, method="ward")
            order = leaves_list(Z)
            clustered = base.iloc[order]

            draw_heatmap_with_controls(
                clustered,
                title="Clustered compliance heatmap (similar departments are adjacent)",
                criteria_order=criteria_order,
                key_prefix="cluster_heat"   # <-- unique keys here
            )

            st.caption("Departments close together have similar compliance patterns.")
