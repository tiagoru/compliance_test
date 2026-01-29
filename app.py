import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Dashboard", layout="wide")
st.title("Compliance Dashboard")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])


def normalize_values(df_long: pd.DataFrame) -> pd.DataFrame:
    """Standardize compliance/goal values and compute a score for KPIs/heatmaps."""
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

    # Optional: parse goal date if present (dd.mm.yyyy)
    df["Goal_date"] = pd.to_datetime(df["Goal_raw"], format="%d.%m.%Y", errors="coerce")

    # Score: Compliant=1, Partial=0.5, else 0
    score_map = {"Compliant": 1.0, "Partial": 0.5}
    df["Score"] = df["Compliance_status"].map(score_map).fillna(0.0)

    return df


def read_excel_two_header(uploaded_file, sheet_name: str) -> pd.DataFrame:
    """
    Reads your Excel where:
    - Row 1 has Department labels repeated: A A B B C C ...
    - Row 2 has field names repeated: Compliant? Goal Compliant? Goal ...
    """
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


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 2-header wide format into long format: Department, Criteria, Compliance_raw, Goal_raw."""
    # Find criteria column (it might be ('Criteria','') or similar)
    criteria_col = None
    for col in df.columns:
        if "criteria" in str(col[0]).lower() or "criteria" in str(col[1]).lower():
            criteria_col = col
            break

    if criteria_col is None:
        raise ValueError("Couldn't find the Criteria column. Make sure the header contains 'Criteria'.")

    criteria_series = df[criteria_col].rename("Criteria")
    rest = df.drop(columns=[criteria_col])

    # Stack departments into rows; the second header level becomes regular columns
    stacked = rest.stack(level=0).reset_index()  # columns: level_0, Department, <fields...>
    stacked = stacked.rename(columns={"level_1": "Department"})

    # Detect the compliance & goal columns from the (former) second header row
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

    # Drop rows where Criteria is blank (sometimes extra empty rows appear)
    out["Criteria"] = out["Criteria"].astype(str).str.strip()
    out = out[out["Criteria"].ne("") & out["Criteria"].ne("nan")]

    return out


if uploaded:
    # Read Excel
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", xls.sheet_names)

    # IMPORTANT CHANGE: read with two header rows
    df_wide = read_excel_two_header(uploaded, sheet)

    st.subheader("Preview (as uploaded)")
    st.dataframe(df_wide.head(15), use_container_width=True)

    # Convert to long + normalize
    try:
        df_long = to_long(df_wide)
    except Exception as e:
        st.error(f"Failed to reshape your Excel file: {e}")
        st.stop()

    df_long = normalize_values(df_long)

    # Sidebar filters
    st.sidebar.header("Filters")
    dept_sel = st.sidebar.multiselect(
        "Department",
        sorted(df_long["Department"].dropna().unique()),
        default=sorted(df_long["Department"].dropna().unique())
    )
    status_sel = st.sidebar.multiselect(
        "Compliance status",
        ["Compliant", "Partial", "Not compliant", "Blank"],
        default=["Compliant", "Partial", "Not compliant"]
    )

    dff = df_long[df_long["Department"].isin(dept_sel)]
    dff = dff[dff["Compliance_status"].isin(status_sel)]

    # KPIs
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    overall = (dff["Score"].mean() * 100) if len(dff) else 0
    c1.metric("Overall compliance", f"{overall:.1f}%")
    c2.metric("Not compliant", int((dff["Compliance_status"] == "Not compliant").sum()))
    c3.metric("Partial", int((dff["Compliance_status"] == "Partial").sum()))
    c4.metric("Missing goals", int((dff["Goal_status"] == "Missing").sum()))

    # Heatmap matrix (pivot + background gradient)
    st.subheader("Department × Criteria matrix (score)")
    pivot = dff.pivot_table(index="Department", columns="Criteria", values="Score", aggfunc="max").fillna(0)

    st.dataframe(
        pivot.style.format("{:.1f}").background_gradient(axis=None),
        use_container_width=True
    )

    # Action list: missing goals
    st.subheader("Action list: Missing goals")
    missing = dff[dff["Goal_status"] == "Missing"][["Department", "Criteria", "Compliance_status", "Goal_status"]]
    st.dataframe(missing, use_container_width=True)

    # Debug table
    with st.expander("See normalized (long) table"):
        st.dataframe(
            dff[["Department", "Criteria", "Compliance_raw", "Goal_raw",
                 "Compliance_status", "Goal_status", "Goal_date", "Score"]],
            use_container_width=True
        )

st.subheader("Goals timeline")

dept_timeline = st.selectbox(
    "Select department for timeline",
    sorted(df_long["Department"].dropna().unique())
)

timeline_df = df_long[
    (df_long["Department"] == dept_timeline) &
    (df_long["Goal_date"].notna()) &
    (df_long["Compliance_status"] != "Compliant")
].copy()

if timeline_df.empty:
    st.info("No dated goals for this department.")
else:
    timeline_df = timeline_df.sort_values("Goal_date")

    fig_timeline = px.scatter(
        timeline_df,
        x="Goal_date",
        y="Criteria",
        color="Compliance_status",
        color_discrete_map={
            "Partial": "#F5C542",        # amber
            "Not compliant": "#E74C3C"   # red
        },
        title=f"{dept_timeline} – Goal Timeline",
        labels={"Goal_date": "Target date"}
    )

    fig_timeline.update_traces(marker=dict(size=12))
    fig_timeline.update_layout(yaxis=dict(autorange="reversed"))

    st.plotly_chart(fig_timeline, use_container_width=True)

st.subheader("Criteria compliance radar")

dept_radar = st.selectbox(
    "Select department for radar",
    sorted(df_long["Department"].dropna().unique()),
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

    fig_radar.update_traces(
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.3)",  # soft green
        line=dict(color="#2ECC71")
    )

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

tab1, tab2, tab3 = st.tabs(["Overview", "Timelines", "Radar"])

with tab1:
    # KPIs + heatmap

with tab2:
    # Timeline code

with tab3:
    # Radar code

