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
    df["Goal_date"] = pd.to_datetime(df["Goal_raw"], format="%d.%m.%Y", errors="coerce")

    score_map = {"Compliant": 1.0, "Partial": 0.5}
    df["Score"] = df["Compliance_status"].map(score_map).fillna(0.0)
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

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Timelines", "Radar", "Cross-department Goals"])


# ----------------------------
# Overview tab
# ----------------------------
with tab1:
    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)
    overall = (dff["Score"].mean() * 100) if len(dff) else 0
    c1.metric("Overall compliance", f"{overall:.1f}%")
    c2.metric("Not compliant", int((dff["Compliance_status"] == "Not compliant").sum()))
    c3.metric("Partial", int((dff["Compliance_status"] == "Partial").sum()))
    c4.metric("Missing goals", int((dff["Goal_status"] == "Missing").sum()))

    st.subheader("Department × Criteria heatmap (score)")
    pivot = dff.pivot_table(index="Department", columns="Criteria", values="Score", aggfunc="max").fillna(0)
    pivot = pivot.reindex(columns=criteria_order)

    if pivot.empty:
        st.info("No data to display (check filters).")
    else:
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            title="Compliance Heatmap (1=Compliant, 0.5=Partial, 0=Not compliant)"
        )
        fig_heat.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Action list: Missing goals")
    missing = dff[dff["Goal_status"] == "Missing"][["Department", "Criteria", "Compliance_status", "Goal_status"]]
    st.dataframe(missing, use_container_width=True)


# ----------------------------
# Timelines tab (single dept)
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
        (df_long["Goal_date"].notna()) &
        (df_long["Compliance_status"] != "Compliant")
    ].copy()

    if timeline_df.empty:
        st.info("No dated goals for this department (or all items are compliant).")
    else:
        today = pd.Timestamp.now().normalize()
        timeline_df["Days_from_now"] = (timeline_df["Goal_date"] - today).dt.days
        timeline_df = timeline_df.sort_values(["Goal_date", "Criteria"])

        fig_timeline = px.scatter(
            timeline_df,
            x="Goal_date",
            y="Criteria",
            color="Compliance_status",
            color_discrete_map={
                "Partial": "#F5C542",
                "Not compliant": "#E74C3C",
                "Blank": "#95A5A6"
            },
            title=f"{dept_timeline} – Goal Dates",
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
            color_discrete_map={
                "Partial": "#F5C542",
                "Not compliant": "#E74C3C",
                "Blank": "#95A5A6"
            },
            title=f"{dept_timeline} – Days Remaining",
            labels={"Days_from_now": "Days from now"}
        )
        fig_days.update_layout(
            yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=criteria_order)
        )
        fig_days.add_vline(x=0, line_width=2, line_dash="dash")
        st.plotly_chart(fig_days, use_container_width=True)


# ----------------------------
# Radar tab
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
# Cross-department goals plot (what you asked for)
# ----------------------------
with tab4:
    st.subheader("Cross-department goals timeline (select departments)")

    # User chooses which departments to plot (can be subset)
    dept_plot = st.multiselect(
        "Choose departments to plot",
        options=all_depts,
        default=dept_sel  # start with current sidebar selection
    )

    # Option to include only non-compliant/partial, or include all
    only_open = st.checkbox("Show only items that are not Compliant (recommended)", value=True)

    cross = df_long[df_long["Department"].isin(dept_plot)].copy()
    cross = cross[cross["Goal_date"].notna()]

    if only_open:
        cross = cross[cross["Compliance_status"] != "Compliant"]

    if cross.empty:
        st.info("No dated goals match your selection.")
    else:
        # Keep criteria order
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
        fig_cross.update_layout(
            yaxis=dict(categoryorder="array", categoryarray=criteria_order),
            legend_title_text="Department"
        )
        st.plotly_chart(fig_cross, use_container_width=True)

        # Optional: show a small table to support the chart
        with st.expander("See underlying rows used in this chart"):
            st.dataframe(
                cross[["Department", "Criteria", "Compliance_status", "Goal_raw", "Goal_date"]]
                .sort_values(["Department", "Goal_date", "Criteria"]),
                use_container_width=True
            )
