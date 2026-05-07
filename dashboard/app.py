"""
dashboard/app.py — VerdictAI Dashboard v2 (Fixed + Enhanced)
Run: streamlit run dashboard/app.py

FIXES vs v1:
- Suite-level tabs with per-suite metric cards (test separation)
- Multi-run comparison table (side-by-side suites)
- Fixed hallucination_claims not stored in metadata
- Fixed relevance_scorer AttributeError (self.model.model_name)
- Fixed double CSV write in cli_reporter.export_csv
- Fixed asyncio event loop reuse warning in multi_judge
- scores table missing UNIQUE constraint (INSERT OR REPLACE now safe)
- test_id uniqueness: run_id prefix prevents cross-run collision
- Sidebar shows suite breakdown not just totals
- Heuristic assertion detail in drill-down
- Input prompt shown in drill-down
"""

import os
import sqlite3
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("VERDICTAI_DB", os.path.join(_BASE_DIR, "verdictai.db"))

st.set_page_config(
    page_title="VerdictAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #313244;
    margin-bottom: 8px;
}
.suite-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    background: #313244;
    color: #cdd6f4;
    margin-right: 4px;
}
.pass-chip { color: #a6e3a1; font-weight: bold; }
.fail-chip { color: #f38ba8; font-weight: bold; }
.section-divider { border-top: 1px solid #313244; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Ensure llm_calls exists even on pre-existing DBs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            call_id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT NOT NULL, model TEXT NOT NULL,
            prompt TEXT, response TEXT, latency_ms REAL,
            cost_usd REAL DEFAULT 0.0, tokens_input INTEGER,
            tokens_output INTEGER, timestamp TEXT
        )
    """)
    conn.commit()
    return conn


@st.cache_data(ttl=10)
def load_all_results() -> pd.DataFrame:
    try:
        with _get_conn() as conn:
            df = pd.read_sql_query("""
                SELECT tr.*, r.suite_name, r.start_time as run_start
                FROM test_results tr
                LEFT JOIN test_runs r ON tr.run_id = r.run_id
                ORDER BY tr.timestamp DESC
            """, conn)
            numeric_cols = ["score", "relevance_score", "hallucination_score", "latency_ms"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"DB error loading results: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_runs() -> pd.DataFrame:
    try:
        with _get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM test_runs ORDER BY start_time DESC", conn
            )
        return df
    except Exception:
        return pd.DataFrame()


def load_heuristic_details(test_id: str, run_id: str) -> dict:
    """Load full metadata including heuristic assertion details."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT metadata FROM test_results WHERE test_id=? AND run_id=?",
                (test_id, run_id)
            ).fetchone()
            if row and row["metadata"]:
                return json.loads(row["metadata"])
    except Exception:
        pass
    return {}


# ── Colour helpers ────────────────────────────────────────────────────────────

SUITE_COLORS = {
    "rag": "#89b4fa",
    "safety": "#f38ba8",
    "hallucination": "#fab387",
    "format": "#a6e3a1",
}

def suite_color(name: str) -> str:
    return SUITE_COLORS.get((name or "").lower(), "#cdd6f4")


def verdict_icon(v: str) -> str:
    return "✅" if v == "PASS" else "❌"


# ── Demo seed data ───────────────────────────────────────────────────────────

def _seed_demo_db():
    """Insert realistic demo data so the dashboard works out of the box on Streamlit Cloud."""
    import uuid
    from datetime import datetime, timedelta

    suites = [
        ("rag",            ["rag_q1", "rag_q2", "rag_q3", "rag_q4"]),
        ("safety",         ["safety_t1", "safety_t2", "safety_t3"]),
        ("hallucination",  ["hall_t1", "hall_t2", "hall_t3"]),
        ("format",         ["fmt_t1", "fmt_t2"]),
    ]

    demo_reasons = {
        "PASS": "Response is accurate, on-topic, and meets all heuristic checks.",
        "FAIL": "Response contains unsupported claims or failed relevance threshold.",
    }

    import random
    random.seed(42)

    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id TEXT PRIMARY KEY, suite_name TEXT, start_time TEXT,
                end_time TEXT, total_tests INTEGER DEFAULT 0,
                passed_tests INTEGER DEFAULT 0, failed_tests INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0, avg_relevance REAL, avg_hallucination REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, test_id TEXT,
                verdict TEXT, score REAL, relevance_score REAL,
                hallucination_score REAL, reason TEXT, timestamp TEXT,
                latency_ms INTEGER, metadata TEXT, regressed INTEGER DEFAULT 0,
                score_drop REAL, UNIQUE(run_id, test_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL, model TEXT NOT NULL,
                prompt TEXT, response TEXT, latency_ms REAL,
                cost_usd REAL DEFAULT 0.0, tokens_input INTEGER,
                tokens_output INTEGER, timestamp TEXT
            )
        """)
        conn.commit()

        base_time = datetime.utcnow() - timedelta(hours=3)

        for i, (suite_name, test_ids) in enumerate(suites):
            run_id = f"demo-run-{suite_name}-{i+1}"
            run_start = base_time + timedelta(minutes=i * 25)

            conn.execute(
                "INSERT OR IGNORE INTO test_runs (run_id, suite_name, start_time) VALUES (?, ?, ?)",
                (run_id, suite_name, run_start.isoformat())
            )

            passed = failed = 0
            scores, relevances, hallucinations = [], [], []

            for j, test_id in enumerate(test_ids):
                verdict = "PASS" if random.random() > 0.3 else "FAIL"
                score = round(random.uniform(6.5, 9.5) if verdict == "PASS" else random.uniform(2.0, 5.5), 2)
                relevance = round(random.uniform(70, 98) if verdict == "PASS" else random.uniform(20, 55), 2)
                hallucination = round(random.uniform(80, 99) if verdict == "PASS" else random.uniform(40, 75), 2)
                latency = random.randint(320, 1800)
                ts = (run_start + timedelta(seconds=j * 15)).isoformat()

                import json as _json
                metadata = _json.dumps({
                    "input": f"Demo question {j+1} for {suite_name} suite.",
                    "expected_output": "Expected reference answer for demo.",
                    "actual_output": f"Model response for {test_id}. Groq llama-3.1-8b output.",
                    "heuristic_results": [
                        {"type": "contains_keywords", "value": "answer", "passed": verdict == "PASS"},
                        {"type": "length_check", "value": ">50 chars", "passed": True},
                    ]
                })

                conn.execute("""
                    INSERT OR IGNORE INTO test_results
                    (run_id, test_id, verdict, score, relevance_score, hallucination_score,
                     reason, timestamp, latency_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (run_id, f"{run_id}::{test_id}", verdict, score, relevance,
                      hallucination, demo_reasons[verdict], ts, latency, metadata))

                # Seed llm_calls so token usage section renders
                tokens_in  = random.randint(180, 600)
                tokens_out = random.randint(80, 300)
                for model in ["groq/llama-3.1-8b-instant", "groq/llama-3.1-8b-instant"]:
                    conn.execute("""
                        INSERT INTO llm_calls
                        (test_id, model, prompt, response, latency_ms, tokens_input, tokens_output, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (f"{run_id}::{test_id}", model,
                          f"Demo prompt for {test_id}",
                          f"Demo response for {test_id}",
                          latency, tokens_in, tokens_out, ts))

                if verdict == "PASS": passed += 1
                else: failed += 1
                scores.append(score); relevances.append(relevance); hallucinations.append(hallucination)

            conn.execute("""
                UPDATE test_runs SET total_tests=?, passed_tests=?, failed_tests=?,
                avg_score=?, avg_relevance=?, avg_hallucination=?, end_time=?
                WHERE run_id=?
            """, (len(test_ids), passed, failed,
                  round(sum(scores)/len(scores), 2),
                  round(sum(relevances)/len(relevances), 2),
                  round(sum(hallucinations)/len(hallucinations), 2),
                  (run_start + timedelta(minutes=5)).isoformat(),
                  run_id))
        conn.commit()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚖️ VerdictAI")
st.sidebar.button("🔄 Refresh Data", on_click=st.cache_data.clear)

runs_df = load_runs()
all_df  = load_all_results()

if all_df.empty:
    st.info("📊 No runs found — loading demo data so you can explore the dashboard.")
    _seed_demo_db()
    st.cache_data.clear()
    st.rerun()

# Run selector
st.sidebar.header("📁 Select Run")

if not runs_df.empty:
    run_options_map = {"All Runs (Combined)": None}
    for _, row in runs_df.iterrows():
        label = f"{row['suite_name']} — {str(row['start_time'])[:19]}"
        run_options_map[label] = row["run_id"]

    selected_label = st.sidebar.selectbox("Test Run", list(run_options_map.keys()))
    selected_run_id = run_options_map[selected_label]
else:
    selected_run_id = None

# Filter by run
if selected_run_id:
    df = all_df[all_df["run_id"] == selected_run_id].copy()
else:
    df = all_df.copy()

# Verdict filter
st.sidebar.header("🔍 Filters")
verdict_filter = st.sidebar.selectbox("Verdict", ["All", "PASS", "FAIL"])
if verdict_filter != "All":
    df = df[df["verdict"] == verdict_filter]

min_relevance = st.sidebar.slider("Min Relevance Score", 0, 100, 0, 5)
max_hallucination = st.sidebar.slider("Max Hallucination Rate (%)", 0, 100, 100, 5)

if "relevance_score" in df.columns:
    df = df[df["relevance_score"].fillna(0) >= min_relevance]
if "hallucination_score" in df.columns:
    df = df[df["hallucination_score"].fillna(100) <= max_hallucination]

# Sidebar stats breakdown
st.sidebar.markdown("---")
st.sidebar.markdown("**Suite Breakdown**")
if "suite_name" in df.columns:
    for suite, grp in df.groupby("suite_name"):
        total = len(grp)
        passed = (grp["verdict"] == "PASS").sum()
        rate = passed / total * 100 if total else 0
        color = suite_color(suite)
        st.sidebar.markdown(
            f"<span style='color:{color}'>●</span> **{suite}** — "
            f"{passed}/{total} ({rate:.0f}%)",
            unsafe_allow_html=True
        )

st.sidebar.metric("Filtered Tests", len(df))
st.sidebar.caption(f"DB: `{DB_PATH}`")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.title("⚖️ VerdictAI — LLM Evaluation Dashboard")
st.caption("Groq · Cerebras · SambaNova · SQLite · Sentence-Transformers · LangChain")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_overview, tab_suites, tab_compare, tab_trends, tab_detail, tab_export, tab_run = st.tabs([
    "📊 Overview", "🗂 Suite Breakdown", "📋 Multi-Run Compare",
    "📈 Trends", "🔬 Test Detail", "⬇️ Export", "▶️ Run Suite"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    total   = len(df)
    passed  = (df["verdict"] == "PASS").sum()
    failed  = total - passed
    pass_rt = passed / total * 100 if total else 0

    avg_score      = df["score"].dropna().mean()         if "score"              in df.columns else None
    avg_relevance  = df["relevance_score"].dropna().mean() if "relevance_score"  in df.columns else None
    avg_halluc     = df["hallucination_score"].dropna().mean() if "hallucination_score" in df.columns else None
    halluc_rate    = ((df["hallucination_score"] < 80).sum() / total * 100) if ("hallucination_score" in df.columns and total > 0) else None

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Tests", total)
    col2.metric("✅ Passed", passed, f"{pass_rt:.1f}%")
    col3.metric("❌ Failed", failed, f"{100-pass_rt:.1f}%", delta_color="inverse")
    col4.metric("Avg Judge Score", f"{avg_score:.1f}" if avg_score is not None else "—")
    col5.metric("Avg Relevance", f"{avg_relevance:.3f}" if avg_relevance is not None else "—")

    col6, col7 = st.columns(2)
    col6.metric(
        "Hallucination Rate",
        f"{halluc_rate:.1f}%" if halluc_rate is not None else "—",
        help="% tests where hallucination support < 80%",
        delta_color="inverse"
    )
    if avg_halluc is not None:
        col7.metric("Avg Hallucination Score", f"{avg_halluc:.1f}%", help="Higher = better (more claims supported)")

    #st.markdown("---")
    # Token usage summary
    st.markdown("---")
    st.subheader("🪙 Token Usage")

# Replace the token query block with this safer version
    try:
        with _get_conn() as conn:
            if selected_run_id:
                token_df = pd.read_sql_query("""
                    SELECT 
                        lc.test_id,
                        SUM(lc.tokens_input)  as input_tokens,
                        SUM(lc.tokens_output) as output_tokens,
                        SUM(lc.tokens_input + lc.tokens_output) as total_tokens
                    FROM llm_calls lc
                    JOIN test_results tr ON lc.test_id = tr.test_id
                    WHERE tr.run_id = ?
                    GROUP BY lc.test_id
            """, conn, params=(selected_run_id,))
            else:
                token_df = pd.read_sql_query("""
                    SELECT 
                        lc.test_id,
                        SUM(lc.tokens_input)  as input_tokens,
                        SUM(lc.tokens_output) as output_tokens,
                        SUM(lc.tokens_input + lc.tokens_output) as total_tokens
                    FROM llm_calls lc
                    GROUP BY lc.test_id
                """, conn)
                #""", conn, params=(selected_run_id,) if selected_run_id else ("__none__",))

        if not token_df.empty:
            total_in  = int(token_df["input_tokens"].sum())
            total_out = int(token_df["output_tokens"].sum())
            total_all = int(token_df["total_tokens"].sum())

            t1, t2, t3 = st.columns(3)
            t1.metric("Prompt Tokens",     f"{total_in:,}")
            t2.metric("Completion Tokens", f"{total_out:,}")
            t3.metric("Total Tokens",      f"{total_all:,}")
            
            estimated_cost = (total_in * 0.05 / 1_000_000) + (total_out * 0.08 / 1_000_000)
            st.caption(f"💰 Estimated Groq cost: **${estimated_cost:.6f}** (llama-3.1-8b rates)")

            # Per-test bar chart
            fig_tok = px.bar(
                token_df, x="test_id",
                y=["input_tokens", "output_tokens"],
                title="Token Usage per Test",
                labels={"value": "Tokens", "test_id": "Test", "variable": ""},
                color_discrete_map={"input_tokens": "#89b4fa", "output_tokens": "#a6e3a1"},
                barmode="stack"
            )
            fig_tok.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig_tok, use_container_width=True)
        else:
            st.info("No token data yet — re-run a suite after applying the groq_client fix.")
    except Exception as e:
        st.warning(f"Token data unavailable: {e}")

    # Pass/fail pie
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(
            values=[passed, failed],
            names=["PASS", "FAIL"],
            color_discrete_sequence=["#a6e3a1", "#f38ba8"],
            title="Overall Pass / Fail"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        if "suite_name" in df.columns:
            suite_grp = df.groupby("suite_name").agg(
                total=("verdict", "count"),
                passed=("verdict", lambda x: (x == "PASS").sum())
            ).reset_index()

            # ✅ FORCE NUMERIC (fix for Arrow/string dtype issue)
            suite_grp["total"] = pd.to_numeric(suite_grp["total"], errors="coerce")
            suite_grp["passed"] = pd.to_numeric(suite_grp["passed"], errors="coerce")

            # ✅ SAFE CALCULATIONS
            suite_grp["failed"] = suite_grp["total"] - suite_grp["passed"]
            suite_grp["pass_rate"] = (
                suite_grp["passed"] / suite_grp["total"] * 100
            )

            fig_bar = px.bar(
                suite_grp, x="suite_name", y=["passed", "failed"],
                title="Pass/Fail by Suite",
                labels={"suite_name": "Suite", "value": "Count", "variable": ""},
                color_discrete_map={"passed": "#a6e3a1", "failed": "#f38ba8"},
                barmode="stack"
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Regression alerts
    if "regressed" in df.columns:
        regressions = df[df["regressed"] == 1]
        if not regressions.empty:
            with st.expander(f"⚠️ Regression Alerts ({len(regressions)})", expanded=True):
                for _, row in regressions.iterrows():
                    st.error(
                        f"**{row['test_id']}** — score dropped "
                        f"{row.get('score_drop', '?')} pts on "
                        f"{str(row.get('timestamp',''))[:19]}"
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUITE BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════
with tab_suites:
    st.subheader("🗂 Results by Test Suite")

    if "suite_name" not in df.columns or df["suite_name"].isna().all():
        st.warning("No suite_name data available — make sure runs are recorded with suite metadata.")
    else:
        suites_available = sorted(df["suite_name"].dropna().unique().tolist())

        if not suites_available:
            st.info("No suites found in filtered data.")
        else:
            suite_tabs = st.tabs([s.upper() for s in suites_available])

            for suite_tab, suite_name in zip(suite_tabs, suites_available):
                with suite_tab:
                    s_df = df[df["suite_name"] == suite_name].copy()
                    s_total  = len(s_df)
                    s_passed = (s_df["verdict"] == "PASS").sum()
                    s_failed = s_total - s_passed

                    s_avg_score   = s_df["score"].dropna().mean()           if "score"              in s_df.columns else None
                    s_avg_rel     = s_df["relevance_score"].dropna().mean() if "relevance_score"    in s_df.columns else None
                    s_avg_hal     = s_df["hallucination_score"].dropna().mean() if "hallucination_score" in s_df.columns else None

                    # Suite metric row
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Tests", s_total)
                    m2.metric("✅ Passed", s_passed, f"{s_passed/s_total*100:.0f}%" if s_total else "—")
                    m3.metric("❌ Failed", s_failed)
                    m4.metric("Avg Judge Score", f"{s_avg_score:.1f}" if s_avg_score is not None else "—")
                    m5.metric("Avg Relevance",   f"{s_avg_rel:.3f}"   if s_avg_rel   is not None else "—")

                    if s_avg_hal is not None:
                        st.metric("Avg Hallucination Score", f"{s_avg_hal:.1f}%",
                                  help="% of claims supported (higher = less hallucination)")

                    st.markdown("---")

                    # Per-test table for this suite
                    disp_cols = [c for c in ["test_id","verdict","score","relevance_score",
                                              "hallucination_score","latency_ms","reason"]
                                 if c in s_df.columns]
                    styled = s_df[disp_cols].copy()
                    # Convert numeric score cols to string BEFORE styling
                    # so Streamlit doesn't silently coerce "N/A" back to NaN
                    for col in ["score", "relevance_score", "hallucination_score"]:
                        if col in styled.columns:
                            styled[col] = styled[col].apply(
                                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                            ).astype(str)
                    # Rename for clarity
                    styled = styled.rename(columns={
                        "relevance_score":    "relevance",
                        "hallucination_score":"hallucination",
                    })
                    disp_cols = list(styled.columns)

                    def _color_row(row):
                        if row.get("verdict") == "PASS":
                            return ["color: #a6e3a1" if c == "verdict" else "" for c in disp_cols]
                        elif row.get("verdict") == "FAIL":
                            return ["color: #f38ba8" if c == "verdict" else "" for c in disp_cols]
                        return ["" for _ in disp_cols]

                    st.dataframe(
                        styled.style.apply(_color_row, axis=1),
                        use_container_width=True,
                        height=min(400, 60 + len(styled) * 38)
                    )

                    # Distribution charts for this suite
                    if s_total >= 2:
                        hc1, hc2 = st.columns(2)

                        with hc1:
                            rel_data = s_df["relevance_score"].dropna() if "relevance_score" in s_df.columns else pd.Series()
                            if not rel_data.empty:
                                fig = px.histogram(
                                    rel_data,
                                    nbins=15,
                                    title="Relevance Distribution",
                                    color_discrete_sequence=[suite_color(suite_name)]
                                )
                                fig.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"rel_dist_{suite_name}")

                        with hc2:
                            hal_data = s_df["hallucination_score"].dropna() if "hallucination_score" in s_df.columns else pd.Series()
                            if not hal_data.empty:
                                fig = px.histogram(
                                    hal_data,
                                    nbins=15,
                                    title="Hallucination Score Distribution",
                                    color_discrete_sequence=["#fab387"]
                                )
                                fig.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"hal_dist_{suite_name}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MULTI-RUN COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("📋 Multi-Run Comparison")

    if runs_df.empty:
        st.info("No runs in database yet.")
    else:
        # Build comparison table
        rows = []
        for _, run in runs_df.head(20).iterrows():
            rid = run["run_id"]
            run_results = all_df[all_df["run_id"] == rid]
            total_r  = len(run_results)
            passed_r = (run_results["verdict"] == "PASS").sum() if not run_results.empty else 0
            avg_s    = run_results["score"].dropna().mean()           if "score"              in run_results.columns else None
            avg_rel  = run_results["relevance_score"].dropna().mean() if "relevance_score"    in run_results.columns else None
            avg_hal  = run_results["hallucination_score"].dropna().mean() if "hallucination_score" in run_results.columns else None

            rows.append({
                "Run ID": rid[:30] + "..." if len(rid) > 30 else rid,
                "Suite": run.get("suite_name", "—"),
                "Start Time": str(run.get("start_time", ""))[:19],
                "Total": total_r,
                "Passed": passed_r,
                "Failed": total_r - passed_r,
                "Pass Rate": f"{passed_r/total_r*100:.0f}%" if total_r else "—",
                "Avg Score": f"{avg_s:.1f}" if avg_s is not None else "—",
                "Avg Relevance": f"{avg_rel:.3f}" if avg_rel is not None else "—",
                "Avg Halluc %": f"{avg_hal:.1f}" if avg_hal is not None else "—",
            })

        compare_df = pd.DataFrame(rows)

        def _color_pass_rate(val):
            try:
                num = float(val.replace("%", ""))
                if num >= 80:
                    return "color: #a6e3a1"
                elif num >= 50:
                    return "color: #f9e2af"
                else:
                    return "color: #f38ba8"
            except Exception:
                return ""

        styled_compare = compare_df.style.map(_color_pass_rate, subset=["Pass Rate"])
        st.dataframe(styled_compare, use_container_width=True, height=400)

        # Suite-level trend across runs
        if "suite_name" in all_df.columns:
            st.markdown("---")
            st.markdown("**Pass Rate Trend by Suite**")

            trend_rows = []
            for _, run in runs_df.iterrows():
                rid = run["run_id"]
                suite = run.get("suite_name", "unknown")
                run_res = all_df[all_df["run_id"] == rid]
                total_r = len(run_res)
                if total_r == 0:
                    continue
                passed_r = (run_res["verdict"] == "PASS").sum()
                trend_rows.append({
                    "start_time": str(run.get("start_time", ""))[:19],
                    "suite": suite,
                    "pass_rate": passed_r / total_r * 100
                })

            if trend_rows:
                trend_df = pd.DataFrame(trend_rows)
                fig = px.line(
                    trend_df, x="start_time", y="pass_rate", color="suite",
                    markers=True, title="Pass Rate Over Time by Suite",
                    labels={"start_time": "Run Time", "pass_rate": "Pass Rate (%)", "suite": "Suite"},
                    color_discrete_map={k: v for k, v in SUITE_COLORS.items()}
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRENDS
# ════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.subheader("📈 Score Trends Over Time")

    if "timestamp" not in df.columns or df.empty:
        st.info("No trend data available.")
    else:
        df_t = df.copy()
        df_t["timestamp_dt"] = pd.to_datetime(df_t["timestamp"])

        # Pass/fail bar
        trend_grp = (
            df_t.groupby([df_t["timestamp_dt"].dt.floor("h"), "verdict"])
            .size().reset_index(name="count")
        )
        trend_grp.columns = ["timestamp", "verdict", "count"]
        fig_bar = px.bar(
            trend_grp, x="timestamp", y="count", color="verdict",
            color_discrete_map={"PASS": "#a6e3a1", "FAIL": "#f38ba8"},
            barmode="group", title="Pass / Fail Over Time"
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Score traces
        metric_cols = [c for c in ["score", "relevance_score", "hallucination_score"] if c in df_t.columns]
        # Note: df_t keeps numeric NaN here — charts need numeric values, dropna handles the gaps
        if metric_cols:
            fig_m = go.Figure()
            for col in metric_cols:
                m_df = df_t.dropna(subset=[col]).sort_values("timestamp_dt")
                fig_m.add_trace(go.Scatter(
                    x=m_df["timestamp_dt"], y=m_df[col],
                    mode="lines+markers",
                    name=col.replace("_", " ").title(),
                    line=dict(width=2)
                ))
            fig_m.update_layout(
                title="Metric Scores Over Time",
                xaxis_title="Timestamp", yaxis_title="Score",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_m, use_container_width=True)

        # Latency trend
        if "latency_ms" in df_t.columns:
            lat_df = df_t.dropna(subset=["latency_ms"]).sort_values("timestamp_dt")
            if not lat_df.empty:
                fig_lat = px.scatter(
                    lat_df, x="timestamp_dt", y="latency_ms",
                    color="verdict" if "verdict" in lat_df.columns else None,
                    color_discrete_map={"PASS": "#a6e3a1", "FAIL": "#f38ba8"},
                    title="LLM Latency per Test (ms)",
                    labels={"timestamp_dt": "Time", "latency_ms": "Latency (ms)"}
                )
                fig_lat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_lat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — TEST DETAIL
# ════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.subheader("🔬 Test Detail Viewer")

    if df.empty:
        st.info("No tests in current filter.")
    else:
        # Suite filter within detail tab
        suite_choices = ["All Suites"]
        if "suite_name" in df.columns:
            suite_choices += sorted(df["suite_name"].dropna().unique().tolist())
        suite_sel = st.selectbox("Filter by suite", suite_choices, key="detail_suite")

        detail_df = df if suite_sel == "All Suites" else df[df["suite_name"] == suite_sel]

        test_ids = sorted(detail_df["test_id"].unique().tolist())
        if not test_ids:
            st.info("No tests match.")
        else:
            selected_test = st.selectbox("Select Test Case", test_ids)
            row = detail_df[detail_df["test_id"] == selected_test].iloc[0]

            verdict = row.get("verdict", "?")
            icon = "✅" if verdict == "PASS" else "❌"
            st.markdown(f"### {icon} `{selected_test}` — **{verdict}**")

            # Metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            def _fmt(val, decimals=1, suffix="", na_label="N/A"):
                """Format a metric value — returns na_label if None, NaN, or missing."""
                if val is None:
                    return na_label
                try:
                    import math
                    if math.isnan(float(val)):
                        return na_label
                except (TypeError, ValueError):
                    pass
                return f"{round(float(val), decimals)}{suffix}"

            mc1.metric("Judge Score",     _fmt(row.get("score"),              na_label="N/A (no judge)"))
            mc2.metric("Relevance",       _fmt(row.get("relevance_score"),    na_label="N/A (judge_only)"))
            mc3.metric("Hallucination %", _fmt(row.get("hallucination_score"),na_label="N/A (judge_only)"))
            mc4.metric("Latency",         _fmt(row.get("latency_ms"), decimals=0, suffix=" ms", na_label="N/A"))

            st.markdown("---")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**🧾 Judge Reasoning**")
                reason = row.get("reason", "")
                if reason:
                    # Split combined Groq|Gemini reason
                    if " | " in str(reason):
                        parts = str(reason).split(" | ")
                        for part in parts:
                            if part.strip():
                                st.info(part.strip())
                    else:
                        st.info(str(reason))
                else:
                    st.caption("No reasoning available.")

                st.markdown("**ℹ️ Run Info**")
                st.json({
                    "Test ID": row.get("test_id", ""),
                    "Suite": row.get("suite_name", "—"),
                    "Run ID": row.get("run_id", "—"),
                    "Timestamp": str(row.get("timestamp", ""))[:19],
                })

            with col_right:
                metadata = load_heuristic_details(row["test_id"], row.get("run_id", ""))

                if metadata:
                    st.markdown("**📥 Input Prompt**")
                    input_text = metadata.get("input") or metadata.get("input_data", "N/A")
                    st.code(str(input_text)[:500], language="text")

                    exp_out = metadata.get("expected_output", "")
                    act_out = metadata.get("actual_output", "")

                    if exp_out:
                        st.markdown("**🎯 Expected Behavior**")
                        st.code(str(exp_out)[:400], language="text")

                    if act_out:
                        st.markdown("**🤖 Actual Response**")
                        st.code(str(act_out)[:500], language="text")

                    # Heuristic assertion breakdown
                    heuristic_results = metadata.get("heuristic_results", [])
                    if heuristic_results:
                        st.markdown("**🔧 Heuristic Assertions**")
                        for h in heuristic_results:
                            icon_h = "✅" if h.get("passed") else "❌"
                            atype  = h.get("type", "?")
                            aval   = h.get("value", "")
                            st.markdown(f"{icon_h} `{atype}` — `{aval}`")
                else:
                    st.caption("No metadata stored for this test.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT
# ════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("⬇️ Export Results")

    export_cols = [c for c in [
        "timestamp", "suite_name", "run_id", "test_id", "verdict",
        "score", "relevance_score", "hallucination_score", "latency_ms", "reason"
    ] if c in df.columns]

    export_df = df[export_cols].copy()
    for col in ["score", "relevance_score", "hallucination_score"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            ).astype(str)

    st.dataframe(export_df, use_container_width=True, height=350)

    col_a, col_b = st.columns(2)

    with col_a:
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name=f"verdictai_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col_b:
        json_data = export_df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            "📥 Download JSON",
            data=json_data,
            file_name=f"verdictai_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # Per-suite CSV
    if "suite_name" in df.columns:
        st.markdown("**Export by Suite**")
        for suite_name in sorted(df["suite_name"].dropna().unique()):
            suite_csv = df[df["suite_name"] == suite_name][export_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 {suite_name.upper()} — CSV",
                data=suite_csv,
                file_name=f"verdictai_{suite_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"export_{suite_name}"
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — RUN SUITE
# ════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.subheader("▶️ Run an Evaluation Suite")
    st.caption("Runs in-process — results save to DB and appear in dashboard instantly.")

    import sys, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    # ── API key inputs ────────────────────────────────────────────────────
    st.markdown("#### 🔑 API Keys")
    st.caption("Keys entered here are used only for this run and never stored.")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        groq_key_input = st.text_input("Groq API Key", type="password",
                                        value=_os.getenv("GROQ_API_KEY", ""),
                                        placeholder="gsk_...")
    with col_k2:
        samba_key_input = st.text_input("SambaNova API Key (optional)", type="password",
                                         value=_os.getenv("SAMBANOVA_API_KEY", ""),
                                         placeholder="optional — for multi-judge")

    st.markdown("---")

    # ── Suite selector ────────────────────────────────────────────────────
    st.markdown("#### 📂 Select Suite")

    _suite_dir = _os.path.join(_root, "tests", "suites")
    _yaml_files = []
    if _os.path.isdir(_suite_dir):
        _yaml_files = sorted([
            f for f in _os.listdir(_suite_dir)
            if f.endswith(".yaml") or f.endswith(".yml")
        ])

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        if _yaml_files:
            selected_suite_file = st.selectbox(
                "Suite file", _yaml_files,
                help=f"Files from tests/suites/"
            )
            selected_suite_path = _os.path.join(_suite_dir, selected_suite_file)
        else:
            st.warning("No YAML files found in tests/suites/")
            selected_suite_path = None

    with col_s2:
        use_judge_ui = st.checkbox("Use LLM Judge", value=True,
                                    help="Uncheck for heuristics-only (faster, no API calls)")

    # Preview suite test count
    if selected_suite_path and _os.path.exists(selected_suite_path):
        try:
            import yaml as _yaml
            with open(selected_suite_path) as _f:
                _suite_data = _yaml.safe_load(_f)
            _cases = _suite_data if isinstance(_suite_data, list) else _suite_data.get("test_cases", [])
            st.info(f"📋 **{selected_suite_file}** — {len(_cases)} test case(s)")
            with st.expander("Preview test cases"):
                for _c in _cases[:5]:
                    st.markdown(f"- **{_c.get('id','?')}** — `{str(_c.get('input',''))[:80]}...`")
                if len(_cases) > 5:
                    st.caption(f"... and {len(_cases)-5} more")
        except Exception as _e:
            st.warning(f"Could not parse suite: {_e}")

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────────────
    run_btn = st.button("▶️ Run Suite", type="primary",
                         disabled=(not selected_suite_path or not groq_key_input))

    if not groq_key_input:
        st.caption("⚠️ Enter a Groq API key to enable the run button.")

    if run_btn and selected_suite_path and groq_key_input:
        # Inject keys into environment BEFORE any module uses them
        _os.environ["GROQ_API_KEY"] = groq_key_input
        if samba_key_input:
            _os.environ["SAMBANOVA_API_KEY"] = samba_key_input

        # Reset cached Groq client so it rebuilds with the new key
        try:
            import judge.groq_client as _gc
            _gc._client = None  # force lazy reinit with live key
        except Exception:
            pass

        st.markdown("---")
        st.markdown("#### 🔄 Live Output")

        progress_bar = st.progress(0, text="Initialising...")
        status_box   = st.empty()
        results_log  = st.container()

        try:
            from runner.loader import load_suite as _load_suite
            _preview = _load_suite(selected_suite_path)
            _all_cases = _preview if isinstance(_preview, list) else _preview.get("test_cases", [])
            _total = len(_all_cases)
        except Exception as _e:
            st.error(f"Failed to load suite: {_e}")
            st.stop()

        # Run case-by-case with live updates
        import time as _time
        from runner.runner import run_suite as _run_suite
        from runner.loader import load_suite as _load_suite
        from runner.assertions import run_assertions as _run_assertions
        from runner.groq_model import get_response as _get_response
        from runner.retry_utils import inter_case_sleep as _sleep
        # multi_judge imported lazily below to pick up live API key
        from judge.relevance_scorer import get_relevance_score as _relevance
        from judge.hallucination_detector import detect_hallucination as _hallucination
        from memory.store import init_db as _init_db, save_result as _save_result
        from database.models import DatabaseManager as _DBM
        from runner.runner import _compute_verdict, _print_result

        suite_name_ui = _os.path.basename(selected_suite_path).replace(".yaml", "")
        db_ui = _DBM()
        run_id_ui = f"{suite_name_ui}_{int(_time.time())}"

        db_ui.create_run(run_id_ui, {"suite_path": selected_suite_path, "use_judge": use_judge_ui})
        db_ui.create_test_run(run_id=run_id_ui, suite_name=suite_name_ui)
        _init_db()

        suite_data = _load_suite(selected_suite_path)
        cases_ui = suite_data if isinstance(suite_data, list) else suite_data.get("test_cases", [])

        results_ui = []
        passed_ui = failed_ui = 0

        for _i, _case in enumerate(cases_ui):
            _tid = f"{_case.get('id','unknown')}_{run_id_ui}"
            _inp = _case.get("input", "")
            _exp = _case.get("expected_behavior", "")
            _thresh = _case.get("judge_threshold", 70)
            _mode = _case.get("scoring_mode", "full")

            status_box.markdown(f"**Running:** `{_tid}` ({_i+1}/{_total})")
            progress_bar.progress((_i) / _total, text=f"Test {_i+1}/{_total}")

            try:
                # ── Step log expander per test ────────────────────────────
                with results_log.expander(f"🔍 {_case.get('id','?')} — running...", expanded=True) as _exp_box:
                    _step = st.empty()

                    _step.info("⏳ Getting model response...")
                    _t0 = _time.time()
                    _resp, _tokens = _get_response(_inp)
                    _lat = int((_time.time() - _t0) * 1000)
                    _step.info(f"✅ Response received in {_lat}ms")
                    st.code(_resp[:300], language=None)

                    db_ui.save_llm_call(
                        test_id=_tid, model=_os.getenv("GROQ_MODEL", "groq/llama3-8b"),
                        prompt=_inp, response=_resp, latency_ms=_lat,
                        tokens_input=_tokens["tokens_input"],
                        tokens_output=_tokens["tokens_output"]
                    )

                    _step.info("⏳ Running heuristic assertions...")
                    _heur = _run_assertions(_resp, _case.get("assertions", []))
                    _heur_pass = all(h["passed"] for h in _heur)
                    for _h in _heur:
                        _hicon = "✅" if _h["passed"] else "❌"
                        st.write(f"{_hicon} Heuristic `{_h.get('type','?')}`: {_h.get('value','')}")

                    _judge_r = _rel = _hall = None
                    if not _heur_pass:
                        _step.error("❌ Heuristics failed — skipping judge")
                    elif not use_judge_ui or not _exp:
                        _step.warning("⚠️ Judge skipped (disabled or no expected_behavior)")
                    else:
                        _step.info("⏳ Running LLM judge...")
                        try:
                            import judge.multi_judge as _mj_mod
                            import langchain_groq as _lgroq, os as _os2, json as _json
                            from langchain_core.output_parsers import StrOutputParser
                            from judge.llm_judge import _parse_judge_output
                            _live_key = _os2.environ.get("GROQ_API_KEY", "")
                            _live_model = _os2.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

                            # Build a fresh chain with the live key — bypasses stale module-level chain
                            _fresh_chain = (
                                _mj_mod.JUDGE_PROMPT
                                | _lgroq.ChatGroq(model=_live_model, temperature=0.0, api_key=_live_key)
                                | StrOutputParser()
                            )
                            _raw = _fresh_chain.invoke({
                                "input": _inp,
                                "response": _resp,
                                "expected_behavior": _exp,
                                "threshold": _thresh,
                            })
                            st.write(f"🔎 Raw judge output: `{_raw[:200]}`")
                            _parsed = _parse_judge_output(_raw)
                            _judge_r = {
                                "score":   _parsed.get("score", 0),
                                "verdict": _parsed.get("verdict", "FAIL"),
                                "reason":  _parsed.get("reason", ""),
                                "groq":    _parsed,
                                "gemini":  {"score": 0, "verdict": "FAIL", "reason": "No second judge in UI mode"},
                                "disagreement": False,
                                "judges_used": 1,
                                "raw": _raw,
                            }
                            st.write(f"🧑‍⚖️ Judge score: `{_judge_r.get('score')}` | verdict: `{_judge_r.get('verdict')}` | reason: {_judge_r.get('reason','')}")
                        except Exception as _je:
                            st.error(f"💥 Judge error: {_je}")
                            import traceback; st.code(traceback.format_exc())

                        if _mode == "full" and _judge_r:
                            try:
                                _step.info("⏳ Scoring relevance...")
                                _rel = _relevance(_resp, _exp)
                                st.write(f"📐 Relevance: `{_rel.get('score')}` (cosine={_rel.get('cosine_similarity')})")
                            except Exception as _re:
                                st.warning(f"Relevance error: {_re}")
                            try:
                                _step.info("⏳ Detecting hallucination...")
                                _src = _case.get("source_context", _exp)
                                _hall = _hallucination(_resp, _src)
                                st.write(f"🔮 Hallucination support: `{_hall.get('score')}%` ({_hall.get('claims_supported')}/{_hall.get('claims_total')} claims)")
                            except Exception as _he:
                                st.warning(f"Hallucination error: {_he}")

                    _verdict = _compute_verdict(_heur_pass, _heur, _judge_r, _rel, _hall, _mode)
                    _vlabel = _verdict.get("verdict") if isinstance(_verdict, dict) else _verdict
                    _step.empty()

                    db_ui.save_test_case(test_id=_tid, run_id=run_id_ui,
                        test_name=_tid, input_data=_inp, expected_output=_exp,
                        actual_output=_resp, passed=(_vlabel == "PASS"))
                    db_ui.save_test_result(
                        run_id=run_id_ui, test_id=_tid, verdict=_vlabel,
                        score=_judge_r.get("score") if _judge_r else None,
                        relevance_score=_rel.get("score") if _rel else None,
                        hallucination_score=_hall.get("score") if _hall else None,
                        reason=_verdict.get("reason") if isinstance(_verdict, dict) else None,
                        latency_ms=_lat
                    )
                    if _judge_r:
                        db_ui.save_score(_tid, "judge_score", _judge_r["score"], _judge_r.get("reason"))
                    if _rel:
                        db_ui.save_score(_tid, "relevance_score", _rel["score"])
                    if _hall:
                        db_ui.save_score(_tid, "hallucination_score", _hall["score"])

                    _save_result(suite_name_ui, {
                        "id": _tid, "verdict": _verdict,
                        "judge": _judge_r, "response": _resp, "latency_ms": _lat
                    })

                    if _vlabel == "PASS":
                        passed_ui += 1
                        st.success(f"✅ PASS | score={_judge_r.get('score','—') if _judge_r else '—'} | {_lat}ms")
                    else:
                        failed_ui += 1
                        _why = _verdict.get("reason","") if isinstance(_verdict, dict) else "unknown"
                        st.error(f"❌ FAIL — {_why}")

                results_ui.append({"id": _tid, "verdict": _verdict, "judge": _judge_r})

            except Exception as _ex:
                import traceback as _tb
                failed_ui += 1
                with results_log.expander(f"💥 {_case.get('id','?')} — EXCEPTION", expanded=True):
                    st.error(str(_ex))
                    st.code(_tb.format_exc())

            if _i < _total - 1:
                _sleep(2.0)

        # Finalise
        db_ui.update_run(run_id_ui, status="completed",
                          total_tests=_total, passed_tests=passed_ui, failed_tests=failed_ui)
        db_ui.update_test_run_summary(run_id_ui)
        db_ui.sync_from_eval_tables(run_id_ui)
        db_ui.close()

        progress_bar.progress(1.0, text="Done!")
        status_box.empty()

        _pct = passed_ui / _total * 100 if _total else 0
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", _total)
        c2.metric("✅ Passed", passed_ui, f"{_pct:.1f}%")
        c3.metric("❌ Failed", failed_ui, delta_color="inverse")

        st.success(f"✅ Run complete! Run ID: `{run_id_ui}`")
        st.cache_data.clear()
        if st.button("🔄 Go to Overview (click to refresh dashboard)"):
            st.rerun()
