"""
Hardware Load Predictor — Streamlit Dashboard
=============================================
Run:  streamlit run dashboard.py
Deps: pip install streamlit requests plotly pandas
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hardware Load Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────

COLOURS = {
    "cpu":     "#ef4444",   # red
    "ram_gb":  "#3b82f6",   # blue
    "ram_pct": "#6366f1",   # indigo
    "net":     "#10b981",   # emerald
    "disk":    "#f59e0b",   # amber
    "biz":     "#8b5cf6",   # violet
    "healthy": "#22c55e",
    "warning": "#f59e0b",
    "error":   "#ef4444",
}

METRIC_LABELS = {
    "predicted_cpu_percent":     "CPU %",
    "predicted_ram_gb":          "RAM GB",
    "predicted_ram_percent":     "RAM %",
    "predicted_network_mbps":    "Network Mbps",
    "predicted_disk_io_percent": "Disk IO %",
}

# ── Sidebar — connection ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("Load Predictor")
    st.markdown("---")
    api_url = st.text_input(
        "Predictor API URL",
        value="http://localhost:8000",
        help="Base URL of the running hardware-load-predictor service",
    )
    prometheus_url = st.text_input(
        "Prometheus URL",
        value="http://localhost:9090",
        help="Used to display live metric graphs",
    )
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if st.button("Refresh now", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.caption("Hardware Load Predictor v2")

BASE = api_url.rstrip("/")
PROM = prometheus_url.rstrip("/")

# ── API helpers ───────────────────────────────────────────────────────────────

def get(path: str, **params) -> Optional[dict | list]:
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=8)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def post(path: str, body: dict) -> Optional[dict | list]:
    try:
        r = requests.post(f"{BASE}{path}", json=body, timeout=30)
        if r.status_code in (200, 201, 202):
            return r.json()
        st.error(f"API error {r.status_code}: {r.text[:300]}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def delete(path: str) -> bool:
    try:
        r = requests.delete(f"{BASE}{path}", timeout=8)
        return r.status_code == 204
    except Exception:
        return False


def prom_query(expr: str, start_minutes: int = 60) -> pd.DataFrame:
    """Query Prometheus range and return a tidy DataFrame."""
    end   = datetime.utcnow()
    start = end - timedelta(minutes=start_minutes)
    try:
        r = requests.get(
            f"{PROM}/api/v1/query_range",
            params={
                "query": expr,
                "start": start.timestamp(),
                "end":   end.timestamp(),
                "step":  "60",
            },
            timeout=8,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        result = r.json().get("data", {}).get("result", [])
        rows = []
        for series in result:
            labels = series.get("metric", {})
            for ts, val in series.get("values", []):
                rows.append({
                    "ts":    datetime.utcfromtimestamp(float(ts)),
                    "value": float(val),
                    **{k: v for k, v in labels.items() if k != "__name__"},
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def api_reachable() -> bool:
    try:
        r = requests.get(f"{BASE}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(0.1)
    st.rerun()

# ── Connection check ──────────────────────────────────────────────────────────

if not api_reachable():
    st.error(f"Cannot reach predictor at **{BASE}**. Is it running?")
    st.code("docker compose up  # or  uvicorn app.main:app --port 8000")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_groups, tab_forecast, tab_metrics, tab_jobs, tab_accuracy = st.tabs([
    "Overview",
    "Server Groups",
    "Forecast",
    "Live Metrics",
    "Training Jobs",
    "Accuracy",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("System Overview")

    groups = get("/groups/") or []
    configs = get("/configs/") or []

    c1, c2, c3, c4 = st.columns(4)
    total_servers = sum(len(g.get("servers", [])) for g in groups)
    active_servers = sum(
        sum(1 for s in g.get("servers", []) if s.get("is_active"))
        for g in groups
    )
    c1.metric("Server Groups", len(groups))
    c2.metric("Total Servers", total_servers)
    c3.metric("Active Servers", active_servers)
    c4.metric("Configs", len(configs))

    st.markdown("---")

    if not groups:
        st.info("No server groups yet. Go to **Server Groups** tab to create one.")
    else:
        for group in groups:
            servers = group.get("servers", [])
            active  = [s for s in servers if s.get("is_active")]

            with st.expander(
                f"**{group['name']}** — {len(active)}/{len(servers)} servers active",
                expanded=True,
            ):
                col_info, col_servers = st.columns([2, 3])

                with col_info:
                    st.markdown(f"**Business metric:** `{group['business_metric_name']}`")
                    st.markdown(f"**Formula:**")
                    st.code(group["business_metric_formula"], language="promql")
                    st.markdown(f"**Prometheus:** `{group['metrics_host']}:{group['metrics_port']}`")
                    if group.get("description"):
                        st.caption(group["description"])

                with col_servers:
                    if servers:
                        df = pd.DataFrame([
                            {
                                "Name":    s["name"],
                                "Host":    f"{s['host']}:{s['port']}",
                                "Active":  "✅" if s["is_active"] else "❌",
                                "Tags":    ", ".join(f"{k}={v}" for k, v in (s.get("tags") or {}).items()),
                            }
                            for s in servers
                        ])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No servers added yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SERVER GROUPS
# ══════════════════════════════════════════════════════════════════════════════

with tab_groups:
    st.header("Server Groups")

    # ── Create group ──────────────────────────────────────────────────────────
    with st.expander("Create new group", expanded=False):
        with st.form("create_group"):
            g_name   = st.text_input("Group name", placeholder="cinema-backend")
            g_desc   = st.text_input("Description (optional)")
            g_bm_name= st.text_input("Business metric name", placeholder="requests_per_minute")
            g_formula= st.text_area("Business metric PromQL",
                                    placeholder='sum(rate(http_auth_responses_codes_total[5m]))',
                                    height=68)
            g_host   = st.text_input("Prometheus host", placeholder="192.168.1.10")
            g_port   = st.number_input("Prometheus port", value=9090, min_value=1, max_value=65535)
            if st.form_submit_button("Create Group", type="primary"):
                if g_name and g_bm_name and g_formula and g_host:
                    result = post("/groups/", {
                        "name": g_name, "description": g_desc or None,
                        "business_metric_name": g_bm_name,
                        "business_metric_formula": g_formula,
                        "metrics_host": g_host, "metrics_port": int(g_port),
                    })
                    if result:
                        st.success(f"Group **{g_name}** created (id={result['id']})")
                        st.rerun()
                else:
                    st.warning("Fill in all required fields.")

    st.markdown("---")

    groups = get("/groups/") or []
    if not groups:
        st.info("No groups yet.")
    else:
        selected_group = st.selectbox(
            "Select group to manage",
            options=groups,
            format_func=lambda g: f"{g['name']} (id={g['id']})",
        )
        g = selected_group

        # ── Manage group ──────────────────────────────────────────────────────
        st.subheader(f"Group: {g['name']}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Servers**")
            servers = g.get("servers", [])
            if servers:
                for s in servers:
                    badge = "OK" if s["is_active"] else "FAIL"
                    st.markdown(f"{badge} **{s['name']}** — `{s['host']}:{s['port']}`")
            else:
                st.caption("No servers yet.")

            # Add server
            with st.form(f"add_server_{g['id']}"):
                st.markdown("**Add server**")
                s_name = st.text_input("Server name", placeholder="gateway")
                s_host = st.text_input("Host (Prometheus)", placeholder="192.168.1.10")
                s_port = st.number_input("Port", value=9090, key=f"sport_{g['id']}")
                s_tags = st.text_input("Tags (key=value, comma-sep)", placeholder="service=gateway,lang=go")
                if st.form_submit_button("Add Server"):
                    tags = {}
                    for pair in s_tags.split(","):
                        if "=" in pair:
                            k, v = pair.strip().split("=", 1)
                            tags[k.strip()] = v.strip()
                    result = post(f"/groups/{g['id']}/servers/", {
                        "name": s_name, "host": s_host,
                        "port": int(s_port), "tags": tags or None,
                    })
                    if result:
                        st.success(f"Server **{s_name}** added.")
                        st.rerun()

        with col_right:
            st.markdown("**Actions**")

            if st.button("🔧 Provision configs", key=f"prov_{g['id']}", use_container_width=True,
                         help="Create a ForecastingConfig for each active server"):
                result = post(f"/groups/{g['id']}/provision", {})
                if result:
                    n = result.get("configs_created", 0)
                    st.success(f"Provisioned **{n}** config(s)." if n else "All servers already provisioned.")

            lookback = st.number_input("Lookback days for training", value=7, min_value=1, max_value=365,
                                       key=f"lb_{g['id']}")
            if st.button("🚀 Train all servers", key=f"train_{g['id']}", use_container_width=True,
                         type="primary"):
                result = post(f"/groups/{g['id']}/train/", {"lookback_days": int(lookback)})
                if result:
                    st.success(f"Submitted **{len(result)}** training job(s).")
                    for job in result:
                        st.caption(f"Job {job['job_id']} — {job['message']}")

            st.markdown("---")
            if st.button("🗑️ Delete group", key=f"del_{g['id']}", use_container_width=True):
                if delete(f"/groups/{g['id']}"):
                    st.success("Deleted.")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    st.header("Cluster Forecast")
    st.markdown("Enter the expected business metric value to predict load across all servers.")

    groups = get("/groups/") or []
    if not groups:
        st.info("Create a server group and train models first.")
    else:
        col_sel, col_inp = st.columns([2, 1])
        with col_sel:
            fc_group = st.selectbox(
                "Server group",
                options=groups,
                format_func=lambda g: g["name"],
                key="fc_group",
            )
        with col_inp:
            biz_val = st.number_input(
                f"Expected {fc_group['business_metric_name']}",
                min_value=0.01, value=10.0, step=0.5,
                key="fc_biz_val",
            )

        if st.button("Run Forecast", type="primary", use_container_width=False):
            with st.spinner("Running inference..."):
                result = post(f"/groups/{fc_group['id']}/forecast/",
                              {"business_metric_value": biz_val})

            if result:
                st.markdown("---")

                # ── Cluster aggregates ──────────────────────────────────────
                st.subheader("Cluster Aggregates")
                ca, cb, cc, cd, ce = st.columns(5)
                ca.metric("Avg CPU",      f"{result['cluster_cpu_avg_percent']:.1f}%",
                          delta_color="inverse")
                cb.metric("Total RAM",    f"{result['cluster_ram_total_gb']:.1f} GB")
                cc.metric("Avg RAM",      f"{result['cluster_ram_avg_percent']:.1f}%",
                          delta_color="inverse")
                cd.metric("Total Net",    f"{result['cluster_network_total_mbps']:.1f} Mbps")
                ce.metric("Avg Disk IO",  f"{result['cluster_disk_avg_io_percent']:.1f}%",
                          delta_color="inverse")

                # ── Per-server breakdown ────────────────────────────────────
                if result["servers"]:
                    st.subheader("Per-Server Predictions")

                    # Gauge chart per server
                    servers_data = result["servers"]
                    n = len(servers_data)
                    cols = st.columns(n)

                    for i, srv in enumerate(servers_data):
                        with cols[i]:
                            st.markdown(f"**{srv['server_name']}**")
                            st.caption(srv["host"])

                            fig = go.Figure()
                            metrics = [
                                ("CPU",        srv["predicted_cpu_percent"],     100, COLOURS["cpu"]),
                                ("RAM %",      srv["predicted_ram_percent"],      100, COLOURS["ram_pct"]),
                                ("Disk IO",    srv["predicted_disk_io_percent"],  100, COLOURS["disk"]),
                            ]
                            for label, val, max_val, colour in metrics:
                                fig.add_trace(go.Indicator(
                                    mode="gauge+number",
                                    value=val,
                                    title={"text": label, "font": {"size": 12}},
                                    gauge={
                                        "axis":  {"range": [0, max_val], "tickfont": {"size": 9}},
                                        "bar":   {"color": colour},
                                        "steps": [
                                            {"range": [0, 60],      "color": "#f0fdf4"},
                                            {"range": [60, 80],     "color": "#fef9c3"},
                                            {"range": [80, max_val],"color": "#fef2f2"},
                                        ],
                                        "threshold": {"line": {"color": "red", "width": 2},
                                                      "thickness": 0.75, "value": 85},
                                    },
                                    domain={"row": metrics.index((label, val, max_val, colour)),
                                            "column": 0},
                                ))

                            fig.update_layout(
                                grid={"rows": 3, "columns": 1, "pattern": "independent"},
                                height=420,
                                margin={"t": 20, "b": 10, "l": 10, "r": 10},
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Absolute values below
                            st.metric("RAM", f"{srv['predicted_ram_gb']:.2f} GB")
                            st.metric("Network", f"{srv['predicted_network_mbps']:.1f} Mbps")

                if result.get("skipped_servers"):
                    st.warning(f"Skipped (no trained model): {', '.join(result['skipped_servers'])}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE METRICS
# ══════════════════════════════════════════════════════════════════════════════

with tab_metrics:
    st.header("Live Metrics from Prometheus")
    st.markdown("Real-time graphs pulled directly from Prometheus.")

    col_window, col_inst = st.columns([1, 2])
    with col_window:
        window_minutes = st.selectbox(
            "Time window", [15, 30, 60, 120, 240], index=2,
            format_func=lambda x: f"Last {x} min",
        )
    with col_inst:
        instance_filter = st.text_input(
            "Instance label filter (leave blank for all)",
            placeholder='e.g. gateway',
            help='Filters node_exporter metrics by instance label',
        )

    inst_selector = f'{{instance="{instance_filter}"}}' if instance_filter else ""

    # ── Business metric ───────────────────────────────────────────────────────
    st.subheader("Business Metric — Requests / min")

    groups = get("/groups/") or []
    if groups:
        bm_group = st.selectbox(
            "Group formula to plot",
            options=groups,
            format_func=lambda g: f"{g['name']} → {g['business_metric_name']}",
            key="bm_group",
        )
        bm_formula = bm_group["business_metric_formula"]
    else:
        bm_formula = st.text_input(
            "PromQL expression",
            value='sum(rate(http_auth_responses_codes_total[5m]))',
        )

    df_bm = prom_query(bm_formula, window_minutes)
    if not df_bm.empty:
        fig_bm = go.Figure()
        for instance, grp in (df_bm.groupby("instance") if "instance" in df_bm.columns
                               else [(None, df_bm)]):
            fig_bm.add_trace(go.Scatter(
                x=grp["ts"], y=grp["value"],
                mode="lines",
                name=str(instance) if instance else "total",
                line={"color": COLOURS["biz"], "width": 2},
                fill="tozeroy",
                fillcolor="rgba(139,92,246,0.1)",
            ))
        fig_bm.update_layout(
            height=220, margin={"t": 10, "b": 30, "l": 50, "r": 10},
            xaxis_title=None, yaxis_title="req/s",
            showlegend=False,
        )
        st.plotly_chart(fig_bm, use_container_width=True)
    else:
        st.warning("No data from Prometheus — check URL or formula.")

    st.markdown("---")

    # ── System metrics ────────────────────────────────────────────────────────
    st.subheader("System Metrics")

    system_queries = [
        ("CPU %",
         f'100 - avg(irate(node_cpu_seconds_total{inst_selector}{{mode="idle"}}[5m])) * 100 by (instance)'
         if instance_filter else
         '100 - avg by (instance)(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100',
         COLOURS["cpu"], "%", [0, 100]),

        ("RAM %",
         f'(1 - node_memory_MemAvailable_bytes{inst_selector} / node_memory_MemTotal_bytes{inst_selector}) * 100'
         if instance_filter else
         '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
         COLOURS["ram_pct"], "%", [0, 100]),

        ("RAM GB",
         f'(node_memory_MemTotal_bytes{inst_selector} - node_memory_MemAvailable_bytes{inst_selector}) / 1073741824'
         if instance_filter else
         '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1073741824',
         COLOURS["ram_gb"], "GB", None),

        ("Network Mbps",
         f'sum(irate(node_network_receive_bytes_total{inst_selector}[5m])) * 8 / 1048576'
         if instance_filter else
         'sum by (instance)(irate(node_network_receive_bytes_total[5m])) * 8 / 1048576',
         COLOURS["net"], "Mbps", None),

        ("Disk IO %",
         f'avg(irate(node_disk_io_time_seconds_total{inst_selector}[5m])) * 100'
         if instance_filter else
         'avg by (instance)(irate(node_disk_io_time_seconds_total[5m])) * 100',
         COLOURS["disk"], "%", [0, 100]),
    ]

    # Two-column grid for system charts
    for row_start in range(0, len(system_queries), 2):
        pair = system_queries[row_start:row_start + 2]
        cols = st.columns(len(pair))
        for col, (title, query, colour, unit, yrange) in zip(cols, pair):
            with col:
                df = prom_query(query, window_minutes)
                fig = go.Figure()
                if not df.empty:
                    group_col = "instance" if "instance" in df.columns else None
                    groups_iter = (df.groupby(group_col) if group_col
                                   else [(None, df)])
                    palette = [colour, "#94a3b8", "#475569"]
                    for idx, (inst, grp) in enumerate(groups_iter):
                        fig.add_trace(go.Scatter(
                            x=grp["ts"], y=grp["value"],
                            mode="lines",
                            name=str(inst) if inst else title,
                            line={"color": palette[idx % len(palette)], "width": 2},
                        ))
                    if yrange:
                        fig.update_yaxes(range=yrange)
                else:
                    fig.add_annotation(text="No data", showarrow=False,
                                       font={"size": 14, "color": "#94a3b8"})

                fig.update_layout(
                    title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                    height=220,
                    margin={"t": 40, "b": 30, "l": 50, "r": 10},
                    xaxis_title=None,
                    yaxis_title=unit,
                    legend={"orientation": "h", "y": -0.2, "font": {"size": 10}},
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRAINING JOBS
# ══════════════════════════════════════════════════════════════════════════════

with tab_jobs:
    st.header("Training Jobs")

    configs = get("/configs/") or []
    if not configs:
        st.info("No configs found. Create a server group and provision first.")
    else:
        selected_cfg = st.selectbox(
            "Config",
            options=configs,
            format_func=lambda c: f"{c['name']} (id={c['id']})",
            key="jobs_cfg",
        )
        cfg_id = selected_cfg["id"]

        col_train, col_models = st.columns(2)

        with col_train:
            st.subheader("Submit Training Job")
            lookback = st.slider("Lookback days", 1, 90, 7, key="jobs_lookback")
            if st.button("🚀 Train", type="primary", key="jobs_train_btn"):
                result = post(f"/configs/{cfg_id}/train/", {"lookback_days": lookback})
                if result:
                    st.success(f"Job submitted — id={result['job_id']}  status={result['status']}")

            st.markdown("---")
            st.subheader("Job History")
            jobs = get(f"/configs/{cfg_id}/train/jobs") or []
            if jobs:
                STATUS_ICON = {"queued": "🕐", "running": "🔄", "done": "✅", "failed": "❌"}
                for job in jobs[:10]:
                    icon = STATUS_ICON.get(job["status"], "❓")
                    dur  = f"  ({job['duration_seconds']:.0f}s)" if job.get("duration_seconds") else ""
                    st.markdown(f"{icon} Job **{job['id']}** — `{job['status']}`{dur}")
                    if job.get("error_message"):
                        with st.expander("Error details"):
                            st.code(job["error_message"])
            else:
                st.caption("No jobs yet.")

        with col_models:
            st.subheader("Trained Models")
            models = get(f"/configs/{cfg_id}/train/models") or []
            if models:
                for model in models[:5]:
                    status_colour = {"ready": "🟢", "training": "🟡",
                                     "failed": "🔴", "pending": "⚪"}.get(model["status"], "⚪")
                    st.markdown(
                        f"{status_colour} **v{model['version']}** — "
                        f"`{model['algorithm']}`  "
                        f"lag={model.get('lag_minutes', '?')} min"
                    )
                    if model.get("metrics") and model["status"] == "ready":
                        m = model["metrics"]
                        cols = st.columns(5)
                        for col, key, label in zip(cols,
                            ["r2_cpu", "r2_ram_gb", "r2_ram_pct", "r2_net", "r2_disk"],
                            ["R² CPU", "R² RAM GB", "R² RAM %", "R² Net", "R² Disk"]):
                            val = m.get(key)
                            col.metric(label, f"{val:.3f}" if val is not None else "—")
                        st.caption(f"MAPE overall: {m.get('mape_overall', '?'):.1f}%")
                    st.markdown("---")
            else:
                st.caption("No models yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ACCURACY
# ══════════════════════════════════════════════════════════════════════════════

with tab_accuracy:
    st.header("Model Accuracy & Drift")

    configs = get("/configs/") or []
    if not configs:
        st.info("No configs found.")
    else:
        acc_cfg = st.selectbox(
            "Config",
            options=configs,
            format_func=lambda c: f"{c['name']} (id={c['id']})",
            key="acc_cfg",
        )
        cfg_id = acc_cfg["id"]

        # Get latest ready model for this config
        models = get(f"/configs/{cfg_id}/train/models") or []
        ready  = [m for m in models if m["status"] == "ready"]

        if not ready:
            st.warning("No trained model for this config yet.")
        else:
            model_id = ready[0]["id"]

            col_status, col_eval = st.columns([2, 1])

            with col_status:
                status = get(f"/models/{model_id}/accuracy")
                if status:
                    health_icon = "✅ Healthy" if status["is_healthy"] else "⚠️ Unhealthy"
                    health_colour = COLOURS["healthy"] if status["is_healthy"] else COLOURS["warning"]
                    st.markdown(
                        f"<h3 style='color:{health_colour}'>{health_icon}</h3>",
                        unsafe_allow_html=True,
                    )
                    if status.get("health_reason"):
                        st.caption(status["health_reason"])

                    st.metric("Evaluations run", status["n_evaluations"])
                    st.metric("Samples evaluated", status["n_samples_total"])

                    ev = status.get("latest_evaluation")
                    if ev:
                        st.markdown("**Latest evaluation**")
                        r2_cols = st.columns(5)
                        for col, key, label in zip(r2_cols,
                            ["r2_cpu", "r2_ram_gb", "r2_ram_pct", "r2_net", "r2_disk"],
                            ["R² CPU", "R² RAM GB", "R² RAM %", "R² Net", "R² Disk"]):
                            val = ev.get(key)
                            colour = (COLOURS["healthy"] if val and val > 0.8
                                      else COLOURS["warning"] if val and val > 0.5
                                      else COLOURS["error"])
                            col.markdown(
                                f"<div style='text-align:center'>"
                                f"<span style='font-size:10px;color:#6b7280'>{label}</span><br>"
                                f"<span style='font-size:20px;font-weight:bold;color:{colour}'>"
                                f"{'—' if val is None else f'{val:.3f}'}</span></div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown("")
                        psi_level = ev.get("psi_level", "unknown")
                        psi_colour = {"stable": COLOURS["healthy"],
                                      "moderate": COLOURS["warning"],
                                      "significant": COLOURS["error"]}.get(psi_level, "#6b7280")
                        st.markdown(
                            f"**Drift (PSI):** "
                            f"<span style='color:{psi_colour};font-weight:bold'>"
                            f"{ev.get('psi_value', '—'):.4f} ({psi_level})</span>",
                            unsafe_allow_html=True,
                        )
                        if ev.get("triggered_retrain"):
                            st.warning("⚡ This evaluation triggered an automatic retraining.")

            with col_eval:
                st.markdown("**Actions**")
                if st.button("Force evaluation", use_container_width=True):
                    with st.spinner("Evaluating..."):
                        result = post(f"/models/{model_id}/accuracy/evaluate", {})
                    if result:
                        st.success("Evaluation complete.")
                        st.rerun()
                    else:
                        st.error("Evaluation failed — need more forecast samples with actuals.")

            # ── History chart ─────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("R² History")
            history = get(f"/models/{model_id}/accuracy/history") or []
            if len(history) >= 2:
                df_hist = pd.DataFrame(history)
                df_hist["evaluated_at"] = pd.to_datetime(df_hist["evaluated_at"])
                df_hist = df_hist.sort_values("evaluated_at")

                fig_r2 = go.Figure()
                r2_cols_map = [
                    ("r2_cpu",     "CPU",     COLOURS["cpu"]),
                    ("r2_ram_gb",  "RAM GB",  COLOURS["ram_gb"]),
                    ("r2_ram_pct", "RAM %",   COLOURS["ram_pct"]),
                    ("r2_net",     "Network", COLOURS["net"]),
                    ("r2_disk",    "Disk IO", COLOURS["disk"]),
                ]
                for key, label, colour in r2_cols_map:
                    if key in df_hist.columns:
                        fig_r2.add_trace(go.Scatter(
                            x=df_hist["evaluated_at"],
                            y=df_hist[key],
                            mode="lines+markers",
                            name=label,
                            line={"color": colour, "width": 2},
                        ))

                fig_r2.add_hline(y=0.85, line_dash="dash", line_color="#94a3b8",
                                 annotation_text="R²=0.85 threshold")
                fig_r2.update_layout(
                    height=300,
                    yaxis={"range": [-0.1, 1.05], "title": "R²"},
                    margin={"t": 10, "b": 30, "l": 50, "r": 10},
                    legend={"orientation": "h", "y": -0.25},
                )
                st.plotly_chart(fig_r2, use_container_width=True)

                # PSI history
                if "psi_value" in df_hist.columns:
                    st.subheader("PSI Drift History")
                    fig_psi = go.Figure()
                    fig_psi.add_trace(go.Scatter(
                        x=df_hist["evaluated_at"], y=df_hist["psi_value"],
                        mode="lines+markers", fill="tozeroy",
                        line={"color": COLOURS["biz"], "width": 2},
                        fillcolor="rgba(139,92,246,0.1)",
                    ))
                    fig_psi.add_hline(y=0.10, line_dash="dot",  line_color=COLOURS["warning"],
                                      annotation_text="moderate")
                    fig_psi.add_hline(y=0.20, line_dash="dash", line_color=COLOURS["error"],
                                      annotation_text="significant → retrain")
                    fig_psi.update_layout(
                        height=200,
                        yaxis_title="PSI",
                        margin={"t": 10, "b": 30, "l": 50, "r": 10},
                    )
                    st.plotly_chart(fig_psi, use_container_width=True)
            else:
                st.info("Run at least 2 evaluations to see history charts.")
