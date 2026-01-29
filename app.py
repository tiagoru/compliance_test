import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compliance Dashboard", layout="wide")
st.title("Compliance Dashboard")

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

def detect_pair_type(col_name: str):
    """Return 'compliance' or 'goal' based on column name, else None."""
    c = col_name.lower()
    if "compliance" in c or "compliant" in c:
        return "compliance"
    if "goal" in c:
        return "goal"
    return None

def extract_department(col_name: str):
    """
    Remove keywords like compliance/goal and separators to get department name.
    Example: 'Dept A - Compliance' -> 'Dept A'
    """
    # remove bracket contents & trim
    s = re.sub(r"\(.*?\)", "", col_name).strip()

    # remove the type words
    s = re.sub(r"(?i)\b(compliance|compliant\??|goal)\b", "", s).strip()

    # cleanup common separators
    s = re.sub(r"[-|:_]+", " ", s).strip()
    s = re.sub(r"\s{2,}", " ", s).strip()

    return s

def normalize_values(df_long: pd.DataFrame) -> pd.DataFrame:
    """Standardize compliance/goal values and compute a score for KPIs/heatmaps."""
    df = df_long.copy()

    # Normalize compliance values
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
        return x  # keep original text if it's something else

    # Normalize goal values
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

    # Score: Compliant=1, Partial=0.5, Not compliant/Blank=0
    score_map = {"Compliant": 1.0, "Partial": 0.5}
    df["Score"] = df["Compliance_status"].map(score_map).fillna(0.0)

    return df

if uploaded:
    # Read Excel
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", xls.sheet_names)
    df_wide = pd.read_excel(uploaded, sheet_name=sheet)

    st.subheader("Preview (as uploaded)")
    st.dataframe(df_wide.head(15), use_container_width=True)

    # Identify criteria column (assume first col)
    criteria_col = df_wide.columns[0]
    other_cols = list(df_wide.columns[1:])

    # Build a mapping: dept -> {'compliance': col, 'goal': col}
    pairs = {}
    for col in other_cols:
        t = detect_pair_type(str(col))
        if not t:
            continue
        dept = extract_department(str(col))
        pairs.setdefault(dept, {})[t] = col

    # Keep only departments that have BOTH columns
    valid_depts = [d for d, p in pairs.items() if "compliance" in p and "goal" in p]

    if not valid_depts:
        st.error(
            "I couldn't detect department column pairs.\n\n"
            "Make sure each department has two columns whose headers include words like "
            "'Compliance/Compliant' and 'Goal'."
        )
        st.stop()

    # Convert wide -> long
    rows = []
    for dept in valid_depts:
        c_col = pairs[dept]["compliance"]
        g_col = pairs[dept]["goal"]
        tmp = df_wide[[criteria_col, c_col, g_col]].copy()
        tmp.columns = ["Criteria", "Compliance_raw", "Goal_raw"]
        tmp["Department"] = dept
        rows.append(tmp)

    df_long = pd.concat(rows, ignore_index=True)
    df_long = normalize_values(df_long)

    # Sidebar filters
    st.sidebar.header("Filters")
    dept_sel = st.sidebar.multiselect("Department", sorted(df_long["Department"].unique()),
                                      default=sorted(df_long["Department"].unique()))
    status_sel = st.sidebar.multiselect("Compliance status",
                                        ["Compliant", "Partial", "Not compliant", "Blank"],
                                        default=["Compliant", "Partial", "Not compliant"])
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

    # Heatmap matrix (simple version using a pivot + st.dataframe coloring)
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

    # Show normalized long table (debug/validation)
    with st.expander("See normalized (long) table"):
        st.dataframe(dff[["Department","Criteria","Compliance_raw","Goal_raw","Compliance_status","Goal_status","Goal_date","Score"]],
                     use_container_width=True)
