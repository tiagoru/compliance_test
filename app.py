import re
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
# Compliance helpers (existing logic)
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
# Heatmap helper (keys fixed)
# ======================================================
def draw_heatmap_with_controls(
    data: pd.DataFrame,
    title: str,
    criteria_order: list[str],
    key_prefix: str,
    show_chunking: bool = True,
    default_chunk: int = 12,
):
    if data.empty:
        st.info("No data to display (check filters).")
        return

    crit_cols = [c for c in criteria_order if c in data.columns]
    other_cols = [c for c in data.columns if c not in crit_cols]
    data = data[crit_cols + other_cols]

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
# NEW: Planner helpers (separate analysis, not using df_long)
# ======================================================
STATUS_COLS_CANON = [
    "Yes",
    "Partial - goal approved",
    "Partial - goal under review",
    "No",
]

def _clean_col(c):
    return "" if pd.isna(c) else str(c).strip()

def _is_year(x: str) -> bool:
    return bool(re.match(r"^20\d{2}$", str(x).strip()))

def _is_quarter(x: str) -> bool:
    return bool(re.match(r"^Q[1-4]$", str(x).strip(), flags=re.I))

def _normalize_status_col(c: str) -> str:
    s = c.strip().lower()
    if s == "yes":
        return "Yes"
    if s == "no":
        return "No"
    if "partial" in s and "approved" in s:
        return "Partial - goal approved"
    if "partial" in s and "under review" in s:
        return "Partial - goal under review"
    return c.strip()

def read_planner_table(uploaded_file, sheet_name: str, start_row: int | None = None) -> pd.DataFrame:
    """
    Read ONLY the planner table. Supports:
    - Multi-header (years + quarters)
    - Flat header (2026 Q1)
    - Optional start_row if planner is below another table (0-based)
    """
    if start_row is None:
        try:
            return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=[0, 1])
        except Exception:
            return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
    else:
        try:
            return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=[start_row, start_row + 1])
        except Exception:
            return pd.read_excel(uploaded_file, sheet_name=sheet_name, header=start_row)

def flatten_planner_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        flat = []
        for a, b in out.columns:
            a = _clean_col(a)
            b = _clean_col(b)

            # Year-quarter -> YYYY-Qn
            if _is_year(a) and _is_quarter(b):
                flat.append(f"{a}-{b.upper()}")
                continue

            # Status columns sometimes appear in header row
            if a and (a.lower() in ["yes", "no"] or "partial" in a.lower()) and not b:
                flat.append(_normalize_status_col(a))
                continue

            # fallback
            if a and b:
                flat.append(f"{a}-{b}")
            else:
                flat.append(a or b)
        out.columns = flat
    else:
        flat = []
        for c in out.columns:
            s = _clean_col(c)

            norm = _normalize_status_col(s)
            if norm in STATUS_COLS_CANON:
                flat.append(norm)
                continue

            m = re.match(r"^\s*(20\d{2})\s*[- ]?\s*(Q[1-4])\s*$", s, flags=re.I)
            if m:
                flat.append(f"{m.group(1)}-{m.group(2).upper()}")
            else:
                flat.append(s)
        out.columns = flat

    return out

def detect_department_column(df: pd.DataFrame) -> str:
    candidates = ["department", "dept", "unit", "name"]
    for c in df.columns:
        if any(k in str(c).lower() for k in candidates):
            return c
    return df.columns[0]

def planner_extract_status_counts(df_flat: pd.DataFrame) -> pd.DataFrame:
    dept_col = detect_department_column(df_flat)
    keep = [dept_col] + [c for c in STATUS_COLS_CANON if c in df_flat.columns]
    if len(keep) == 1:
        raise ValueError("No status count columns found (Yes / Partial... / No).")

    out = df_flat[keep].copy().rename(columns={dept_col: "Department"})
    for c in keep[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out

def planner_extract_quarters_long(df_flat: pd.DataFrame) -> pd.DataFrame:
    dept_col = detect_department_column(df_flat)
    quarter_cols = [c for c in df_flat.columns if re.match(r"^20\d{2}-Q[1-4]$", str(c))]
    if not quarter_cols:
        raise ValueError("No quarter columns found (expected 2026-Q1 ... 2029-Q4).")

    out = df_flat.melt(
        id_vars=[dept_col],
        value_vars=quarter_cols,
        var_name="Period",
        value_name="Count",
    ).rename(columns={dept_col: "Department"})

    out["Count"] = pd.to_numeric(out["Count"], errors="coerce").fillna(0).astype(int)

    def sort_key(p):
        m = re.match(r"^(20\d{2})-Q([1-4])$", str(p))
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)

    period_order = sorted(out["Period"].unique(), key=sort_key)
    out["Period"] = pd.Categorical(out["Period"], categories=period_order, ordered=True)
    return out


# ======================================================
# App start
# ======================================================
if not uploaded:
    st.info("Upload an Excel file to begin.")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet = st.selectbox("Select compliance sheet", xls.sheet_names, key="main_sheet")

df_wide = read_excel_two_header(uploaded, sheet)
st.subheader("Preview (compliance sheet)")
st.dataframe(df_wide.head(15), use_container_width=True)

try:
    df_long, criteria_order = to_long_with_order(df_wide)
except Exception as e:
    st.error(f"Failed to reshape compliance table: {e}")
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
dept_sel = st.sidebar.multiselect("Department", all_depts, default=all_depts, key="flt_depts")

status_options = ["Compliant", "Partial", "Not compliant", "Blank"]
status_sel = st.sidebar.multiselect(
    "Compliance status",
    status_options,
    default=["Compliant", "Partial", "Not compliant"],
    key="flt_status"
)

dff = df_long[df_long["Department"].isin(dept_sel)]
dff = dff[dff["Compliance_status"].isin(status_sel)]

# Tabs (existing + NEW separate planner tab)
tab_exec, tab_overview, tab_timelines, tab_radar, tab_cross, tab_cluster, tab_planner = st.tabs(
    ["Executive Summary", "Overview", "Timelines", "Radar", "Cross-department Goals", "Clustered Heatmap", "Counts & Planner (separate)"]
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

# ======================================================
# Overview
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

# ======================================================
# Timelines
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
# Cross-department Goals
# ======================================================
with tab_cross:
    st.subheader("Cross-department goals timeline (select departments)")

    dept_plot = st.multiselect("Choose departments to plot", options=all_depts, default=dept_sel, key="cross_depts")
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
# Clustered Heatmap
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
                key_prefix="cluster_heat"
            )

# ======================================================
# NEW TAB: Counts & Planner (separate table)
# ======================================================
with tab_planner:
    st.subheader("Counts & Planner (separate table)")
    st.info("This tab reads the new counts/planner table separately and does NOT use the compliance reshaping.")

    planner_sheet = st.selectbox("Select sheet that contains the counts/planner table", xls.sheet_names, key="planner_sheet_sel")

    use_start = st.checkbox("Planner starts lower in the sheet (set header row manually)", value=False, key="planner_use_start")
    start_row = None
    if use_start:
        start_row = st.number_input("Planner header row (0-based)", min_value=0, value=0, step=1, key="planner_start_row")

    try:
        df_pl_raw = read_planner_table(uploaded, planner_sheet, start_row=start_row)
        df_pl_flat = flatten_planner_columns(df_pl_raw)
    except Exception as e:
        st.error(f"Could not read counts/planner table: {e}")
        st.stop()

    st.markdown("### Preview (counts/planner table)")
    st.dataframe(df_pl_flat.head(30), use_container_width=True)

    t1, t2 = st.tabs(["Status counts", "Quarter planner"])

    # Status counts
    with t1:
        try:
            status_df = planner_extract_status_counts(df_pl_flat)
        except Exception as e:
            st.error(str(e))
            st.stop()

        match_mode = st.checkbox("Filter by selected departments (if names match)", value=True, key="planner_match_depts")
        if match_mode:
            status_df = status_df[status_df["Department"].isin(dept_sel)].copy()

        if status_df.empty:
            st.info("No departments matched your selection (or the table has no rows).")
        else:
            st.dataframe(status_df, use_container_width=True)

            plot_long = status_df.melt(id_vars=["Department"], var_name="Status", value_name="Count")
            fig = px.bar(plot_long, x="Department", y="Count", color="Status", barmode="stack",
                         title="Counts by status (per department)")
            st.plotly_chart(fig, use_container_width=True)

    # Quarter planner
    with t2:
        try:
            plan_long = planner_extract_quarters_long(df_pl_flat)
        except Exception as e:
            st.error(str(e))
            st.stop()

        match_mode2 = st.checkbox("Filter by selected departments (if names match)", value=True, key="planner_match_depts2")
        if match_mode2:
            plan_long = plan_long[plan_long["Department"].isin(dept_sel)].copy()

        if plan_long.empty:
            st.info("No departments matched your selection (or the table has no rows).")
        else:
            piv = plan_long.pivot_table(index="Department", columns="Period", values="Count", aggfunc="sum").fillna(0).astype(int)
            fig_hm = px.imshow(piv, aspect="auto", title="Quarter planner heatmap (Department × Quarter)")
            fig_hm.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_hm, use_container_width=True)

            dept_plot2 = st.multiselect(
                "Departments to compare",
                options=sorted(plan_long["Department"].unique()),
                default=sorted(plan_long["Department"].unique())[:5],
                key="planner_compare_depts"
            )
            comp = plan_long[plan_long["Department"].isin(dept_plot2)].copy()
            fig_line = px.line(comp.sort_values("Period"), x="Period", y="Count", color="Department", markers=True,
                               title="Quarterly plan comparison")
            st.plotly_chart(fig_line, use_container_width=True)
