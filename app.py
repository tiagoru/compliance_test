import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Compliance Dashboard", layout="wide")
st.title("Compliance Dashboard")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])


# ----------------------------
# Helpers
# ----------------------------
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

    # Parse dd.mm.yyyy dates if present
    df["Goal_date"] = pd.to_datetime(df["Goal_raw"], format="%d.%m.%Y", errors="coerce")

    # Score for charts
    score_map = {"Compliant": 1.0, "Partial": 0.5}
    df["Score"] = df["Compliance_status"].map(score_map).fillna(0.0)

    # Convenience flags
    df["Is_open_item"] = df["Compliance_status"].isin(["Partial", "Not compliant"])
    df["Has_goal_date"] = df["Goal_date"].notna()
    df["Has_missing_goal"] = df["Goal_status"].eq("Missing")

    return df


def read_excel_two_header(uploaded_file, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=[0, 1])

    # Forward-fill department names (merged cells often create blanks)
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
    # Find criteria column (it might be ('Criteria','') or similar)
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
    stacked = rest.stack(level=0).reset_index()
    stacked = stacked.rename(columns={"level_1": "Department"})

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


# ----------------------------
# App
# ----------------------------
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

# Dates (use local date; avoid timezone issues in pandas/plotly)
today = pd.Timestamp.now(tz="Europe/Berlin").normalize().tz_localize(None)
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
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Timelines", "Radar", "Cross-department Goals"])

# ----------------------------
# OVERVIEW: KPIs + Heatmap + Leaderboard + Goal Density + Action Lists
# ----------------------------
with tab1:
    st.subheader("Overview")

    # KPI cards (high impact)
    open_items = dff[dff["Is_open_item"]]
    overdue = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] < today)]
    due_30 = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d30)]
    due_60 = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] > d30) & (open_items["Goal_date"] <= d60)]
    due_90 = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] > d60) & (open_items["Goal_date"] <= d90)]

    c1, c2, c3, c4, c5 = st.columns(5)
    overall = (dff["Score"].mean() * 100) if len(dff) else 0
    c1.metric("Overall compliance", f"{overall:.1f}%")
    c2.metric("Open items", int(open_items.shape[0]))
    c3.metric("Overdue goals", int(overdue.shape[0]))
    c4.metric("Due in 30 days", int(due_30.shape[0]))
    c5.metric("Missing goals", int(dff[dff["Has_missing_goal"]].shape[0]))

    # Heatmap with row/column totals
    st.subheader("Department × Criteria heatmap (score) + totals")

    pivot = dff.pivot_table(index="Department", columns="Criteria", values="Score", aggfunc="max").fillna(0)
    pivot = pivot.reindex(columns=criteria_order)

    if pivot.empty:
        st.info("No data to display (check filters).")
    else:
        pivot_with_totals = pivot.copy()
        pivot_with_totals["Dept Avg"] = pivot.mean(axis=1)

        criteria_avg = pivot.mean(axis=0)
        criteria_avg["Dept Avg"] = pivot_with_totals["Dept Avg"].mean() if len(pivot_with_totals) else 0.0
        pivot_with_totals.loc["Criteria Avg"] = criteria_avg

        fig_heat = px.imshow(
            pivot_with_totals,
            aspect="auto",
            title="Compliance Heatmap (1=Compliant, 0.5=Partial, 0=Not compliant). Includes Dept Avg + Criteria Avg."
        )
        fig_heat.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Department leaderboard (comparison table)
    st.subheader("Department leaderboard (comparison)")

    def _dept_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
        grp = df_in.groupby("Department", dropna=True)

        out = grp.agg(
            Criteria_count=("Criteria", "nunique"),
            Compliance_pct=("Score", lambda s: round(float(s.mean() * 100), 1) if len(s) else 0.0),
            Open_items=("Is_open_item", "sum"),
            Missing_goals=("Has_missing_goal", "sum"),
            Dated_goals=("Has_goal_date", "sum"),
        ).reset_index()

        # Overdue and due soon (open items only)
        def count_where(sub, mask):
            return int(mask.loc[sub.index].sum())

        mask_open = df_in["Is_open_item"]
        mask_overdue = mask_open & df_in["Has_goal_date"] & (df_in["Goal_date"] < today)
        mask_due30 = mask_open & df_in["Has_goal_date"] & (df_in["Goal_date"] >= today) & (df_in["Goal_date"] <= d30)

        out["Overdue"] = out["Department"].apply(
            lambda d: count_where(df_in[df_in["Department"] == d], mask_overdue)
        )
        out["Due_30d"] = out["Department"].apply(
            lambda d: count_where(df_in[df_in["Department"] == d], mask_due30)
        )

        # Avg days to goal for open items with dates
        def avg_days(d):
            sub = df_in[(df_in["Department"] == d) & mask_open & df_in["Has_goal_date"]]
            if sub.empty:
                return None
            return float((sub["Goal_date"] - today).dt.days.mean())

        out["Avg_days_to_goal_open"] = out["Department"].apply(avg_days).round(1)

        # Sort: most overdue, then lowest compliance
        out = out.sort_values(["Overdue", "Compliance_pct"], ascending=[False, True])
        return out

    leaderboard = _dept_metrics(dff)
    st.dataframe(leaderboard, use_container_width=True)

    # Goal density over time (compare departments)
    st.subheader("Goal density over time (compare departments)")

    goal_density = open_items[open_items["Has_goal_date"]].copy()
    if goal_density.empty:
        st.info("No dated goals for open items (Partial/Not compliant) in the current selection.")
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
            title="Number of open-item goals by month (stacked by department)",
            labels={"Goal_count": "Goals due"}
        )
        st.plotly_chart(fig_density, use_container_width=True)

    # Action lists split: Missing / Overdue / Due soon
    st.subheader("Action lists")

    t_missing, t_overdue, t_due = st.tabs(["Missing goals", "Overdue", "Due soon (30/60/90)"])

    with t_missing:
        miss_tbl = dff[dff["Has_missing_goal"]][["Department", "Criteria", "Compliance_status", "Goal_raw"]]
        st.dataframe(miss_tbl, use_container_width=True)

    with t_overdue:
        overdue_tbl = overdue[["Department", "Criteria", "Compliance_status", "Goal_date", "Goal_raw"]].sort_values(["Department", "Goal_date"])
        st.dataframe(overdue_tbl, use_container_width=True)

    with t_due:
        window = st.radio("Choose window", ["30 days", "60 days", "90 days"], horizontal=True)
        if window == "30 days":
            due_tbl = due_30
            label_end = d30
        elif window == "60 days":
            due_tbl = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d60)]
            label_end = d60
        else:
            due_tbl = open_items[open_items["Has_goal_date"] & (open_items["Goal_date"] >= today) & (open_items["Goal_date"] <= d90)]
            label_end = d90

        st.caption(f"Showing open items with goal dates between {today.date()} and {label_end.date()}.")
        due_tbl = due_tbl.copy()
        due_tbl["Days_from_now"] = (due_tbl["Goal_date"] - today).dt.days
        st.dataframe(
            due_tbl[["Department", "Criteria", "Compliance_status", "Goal_date", "Days_from_now"]]
            .sort_values(["Goal_date", "Department"]),
            use_container_width=True
        )

    with st.expander("See normalized (long) table"):
        st.dataframe(
            dff[["Department", "Criteria", "Compliance_raw", "Goal_raw",
                 "Compliance_status", "Goal_status", "Goal_date", "Score"]],
            use_container_width=True
        )

# ----------------------------
# TIMELINES: single dept timeline + days-from-now bar
# ----------------------------
with tab2:
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
        st.info("No dated goals for open items (Partial/Not compliant) in this department.")
    else:
        timeline_df["Days_from_now"] = (timeline_df["Goal_date"] - today).dt.days
        timeline_df = timeline_df.sort_values(["Goal_date", "Criteria"])

        fig_timeline = px.scatter(
            timeline_df,
            x="Goal_date",
            y="Criteria",
            color="Compliance_status",
            color_discrete_map={"Partial": "#F5C542", "Not compliant": "#E74C3C", "Blank": "#95A5A6"},
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
            color_discrete_map={"Partial": "#F5C542", "Not compliant": "#E74C3C", "Blank": "#95A5A6"},
            title=f"{dept_timeline} – Days Remaining (open items only)",
            labels={"Days_from_now": "Days from now"}
        )
        fig_days.update_layout(yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=criteria_order))
        fig_days.add_vline(x=0, line_width=2, line_dash="dash")
        st.plotly_chart(fig_days, use_container_width=True)

# ----------------------------
# RADAR: department compliance by criteria
# ----------------------------
with tab3:
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
            title=f"{dept_radar} – Compliance by Criteria (ordered)"
        )
        fig_radar.update_traces(fill="toself")
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    tickvals=[0, 0.5, 1],
                    ticktext=["Not compliant", "Partial", "Compliant"],
                    visible=True
                )
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ----------------------------
# CROSS-DEPARTMENT GOALS: X=goal date, Y=criteria, departments together, user picks which to plot
# ----------------------------
with tab4:
    st.subheader("Cross-department goals timeline (select departments)")

    dept_plot = st.multiselect(
        "Choose departments to plot",
        options=all_depts,
        default=dept_sel
    )
    only_open = st.checkbox("Show only open items (Partial / Not compliant)", value=True)

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

        with st.expander("See underlying rows used in this chart"):
            st.dataframe(
                cross[["Department", "Criteria", "Compliance_status", "Goal_raw", "Goal_date"]]
                .sort_values(["Department", "Goal_date", "Criteria"]),
                use_container_width=True
            )
