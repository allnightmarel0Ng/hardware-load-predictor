import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Прогнозирование нагрузки",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


COLOURS = {
    "cpu":     "#ef4444",
    "ram_gb":  "#3b82f6",
    "ram_pct": "#6366f1",
    "net":     "#10b981",
    "disk":    "#f59e0b",
    "biz":     "#8b5cf6",
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


with st.sidebar:
    st.title("Прогнозирование нагрузки")
    st.markdown("---")
    api_url = st.text_input(
        "Predictor API URL",
        value="http://localhost:8000",
        help="Базовый URL запущенного сервиса прогнозирования",
    )
    prometheus_url = st.text_input(
        "Prometheus URL",
        value="http://localhost:9090",
        help="Used to display live metric graphs",
    )
    auto_refresh = st.toggle("Авто-обновление (30 с)", value=False)
    if st.button("Обновить", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.caption("Система прогнозирования нагрузки v2")

BASE = api_url.rstrip("/")
PROM = prometheus_url.rstrip("/")


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


if auto_refresh:
    time.sleep(0.1)
    st.rerun()


if not api_reachable():
    st.error(f"Не удаётся подключиться к сервису по адресу {BASE}. Проверьте, запущен ли сервис.")
    st.code("docker compose up  # или  uvicorn app.main:app --port 8000")
    st.stop()


tab_overview, tab_groups, tab_forecast, tab_metrics, tab_jobs, tab_accuracy = st.tabs([
    "Обзор",
    "Группы серверов",
    "Прогноз",
    "Живые метрики",
    "Задания обучения",
    "Точность",
])


with tab_overview:
    st.header("Обзор системы")

    groups = get("/groups/") or []
    configs = get("/configs/") or []

    c1, c2, c3, c4 = st.columns(4)
    total_servers = sum(len(g.get("servers", [])) for g in groups)
    active_servers = sum(
        sum(1 for s in g.get("servers", []) if s.get("is_active"))
        for g in groups
    )
    c1.metric("Групп серверов", len(groups))
    c2.metric("Серверов всего", total_servers)
    c3.metric("Активных серверов", active_servers)
    c4.metric("Конфигураций", len(configs))

    st.markdown("---")

    if not groups:
        st.info("Нет групп серверов. Перейдите на вкладку **Группы серверов** для создания.")
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
                                "Активен": "Да" if s["is_active"] else "Нет",
                                "Tags":    ", ".join(f"{k}={v}" for k, v in (s.get("tags") or {}).items()),
                            }
                            for s in servers
                        ])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.info("Серверы не добавлены.")


with tab_groups:
    st.header("Групп серверов")

    with st.expander("Создать группу", expanded=False):
        with st.form("create_group"):
            g_name   = st.text_input("Название группы", placeholder="cinema-backend")
            g_desc   = st.text_input("Описание (необязательно)")
            g_bm_name= st.text_input("Название бизнес-метрики", placeholder="requests_per_minute")
            g_formula= st.text_area("PromQL бизнес-метрики",
                                    placeholder='sum(rate(http_auth_responses_codes_total[5m]))',
                                    height=68)
            g_host   = st.text_input("Хост Prometheus", placeholder="192.168.1.10")
            g_port   = st.number_input("Порт Prometheus", value=9090, min_value=1, max_value=65535)
            if st.form_submit_button("Создать группу", type="primary"):
                if g_name and g_bm_name and g_formula and g_host:
                    result = post("/groups/", {
                        "name": g_name, "description": g_desc or None,
                        "business_metric_name": g_bm_name,
                        "business_metric_formula": g_formula,
                        "metrics_host": g_host, "metrics_port": int(g_port),
                    })
                    if result:
                        st.success(f"Группа **{g_name}** создана (id={result['id']})")
                        st.rerun()
                else:
                    st.warning("Заполните все обязательные поля.")

    st.markdown("---")

    groups = get("/groups/") or []
    if not groups:
        st.info("Нет групп.")
    else:
        selected_group = st.selectbox(
            "Выберите группу для управления",
            options=groups,
            format_func=lambda g: f"{g['name']} (id={g['id']})",
        )
        g = selected_group

        st.subheader(f"Group: {g['name']}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Servers**")
            servers = g.get("servers", [])
            if servers:
                for s in servers:
                    badge = "[активен]" if s["is_active"] else "[неактивен]"
                    st.markdown(f"{badge} **{s['name']}** — `{s['host']}:{s['port']}`")
            else:
                st.caption("Серверов нет.")

            with st.form(f"add_server_{g['id']}"):
                st.markdown("**Add server**")
                s_name = st.text_input("Имя сервера", placeholder="gateway")
                s_host = st.text_input("Хост (Prometheus)", placeholder="192.168.1.10")
                s_port = st.number_input("Port", value=9090, key=f"sport_{g['id']}")
                s_tags = st.text_input("Теги (key=value через запятую)", placeholder="service=gateway,lang=go")
                if st.form_submit_button("Добавить сервер"):
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
                        st.success(f"Сервер **{s_name}** добавлен.")
                        st.rerun()

        with col_right:
            st.markdown("**Действия**")

            if st.button("Создать конфиги", key=f"prov_{g['id']}", use_container_width=True,
                         help="Создать ForecastingConfig для каждого активного сервера"):
                result = post(f"/groups/{g['id']}/provision", {})
                if result:
                    n = result.get("configs_created", 0)
                    st.success(f"Создано конфигураций: {n}." if n else "Все серверы уже сконфигурированы.")

            lookback = st.number_input("Дней истории для обучения", value=7, min_value=1, max_value=365,
                                       key=f"lb_{g['id']}")
            if st.button("Обучить все серверы", key=f"train_{g['id']}", use_container_width=True,
                         type="primary"):
                result = post(f"/groups/{g['id']}/train/", {"lookback_days": int(lookback)})
                if result:
                    st.success(f"Submitted **{len(result)}** training job(s).")
                    for job in result:
                        st.caption(f"Job {job['job_id']} — {job['message']}")

            st.markdown("---")
            if st.button("Удалить группу", key=f"del_{g['id']}", use_container_width=True):
                if delete(f"/groups/{g['id']}"):
                    st.success("Удалено.")
                    st.rerun()


with tab_forecast:
    st.header("Прогноз нагрузки кластера")
    st.markdown("Введите ожидаемое значение бизнес-метрики для прогноза нагрузки на все серверы.")

    groups = get("/groups/") or []
    if not groups:
        st.info("Сначала создайте группу серверов и обучите модели.")
    else:
        col_sel, col_inp = st.columns([2, 1])
        with col_sel:
            fc_group = st.selectbox(
                "Группа серверов",
                options=groups,
                format_func=lambda g: g["name"],
                key="fc_group",
            )
        with col_inp:
            biz_val = st.number_input(
                "Значение бизнес-метрики",
                min_value=0.01, value=10.0, step=0.5,
                key="fc_biz_val",
            )

        if st.button("Запустить прогноз", type="primary", use_container_width=False):
            with st.spinner("Выполняется прогноз..."):
                result = post(f"/groups/{fc_group['id']}/forecast/",
                              {"business_metric_value": biz_val})

            if result:
                st.markdown("---")

                st.subheader("Агрегаты кластера")
                ca, cb, cc, cd, ce = st.columns(5)
                ca.metric("CPU (среднее)",    f"{result['cluster_cpu_avg_percent']:.1f}%")
                cb.metric("RAM (сумма)",       f"{result['cluster_ram_total_gb']:.2f} ГБ")
                cc.metric("RAM (среднее)",     f"{result['cluster_ram_avg_percent']:.1f}%")
                cd.metric("Сеть (сумма)",      f"{result['cluster_network_total_mbps']:.1f} Мбит/с")
                ce.metric("Диск IO (среднее)", f"{result['cluster_disk_avg_io_percent']:.1f}%")

                if result["servers"]:
                    st.subheader("Прогноз по серверам")
                    servers_data = result["servers"]
                    n_srv = len(servers_data)
                    cols = st.columns(n_srv)

                    for i, srv in enumerate(servers_data):
                        with cols[i]:
                            st.markdown(f"#### {srv['server_name']}")
                            st.caption(srv["host"])

                            cpu  = srv["predicted_cpu_percent"]
                            ramp = srv["predicted_ram_percent"]
                            ramg = srv["predicted_ram_gb"]
                            net  = srv["predicted_network_mbps"]
                            disk = srv["predicted_disk_io_percent"]

                            def colour(v):
                                if v >= 85: return "#dc2626"
                                if v >= 60: return "#d97706"
                                return "#16a34a"

                            def bar(v, mx=100):
                                pct = min(100, v / mx * 100)
                                return (
                                    f'<div style="background:#e5e7eb;border-radius:4px;'
                                    f'height:10px;margin:2px 0 8px 0">'
                                    f'<div style="width:{pct:.1f}%;background:{colour(v)};'
                                    f'border-radius:4px;height:10px"></div></div>'
                                )

                            st.markdown(
                                f'**Загрузка CPU:**&nbsp; '
                                f'<span style="color:{colour(cpu)};font-size:1.4em;font-weight:700">'
                                f'{cpu:.1f}%</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(bar(cpu), unsafe_allow_html=True)

                            st.markdown(
                                f'**Оперативная память:**&nbsp; '
                                f'<span style="color:{colour(ramp)};font-size:1.4em;font-weight:700">'
                                f'{ramp:.1f}%</span>&nbsp; ({ramg:.2f} ГБ)',
                                unsafe_allow_html=True,
                            )
                            st.markdown(bar(ramp), unsafe_allow_html=True)

                            st.markdown(
                                f'**Сетевой трафик:**&nbsp; '
                                f'<span style="color:#1d4ed8;font-size:1.4em;font-weight:700">'
                                f'{net:.1f} Мбит/с</span>',
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                f'**Диск IO:**&nbsp; '
                                f'<span style="color:{colour(disk)};font-size:1.4em;font-weight:700">'
                                f'{disk:.1f}%</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(bar(disk), unsafe_allow_html=True)

                            cfg_models = get(f"/configs/{srv['config_id']}/models/") or []
                            if cfg_models:
                                m = cfg_models[0]
                                pt = m.get("parameters", {}).get("per_target", {})
                                mtr = m.get("metrics", {})
                                st.markdown("---")
                                st.caption("Качество моделей (тестовая выборка)")
                                rows = []
                                for key, lbl in [("cpu","CPU"),("ram_gb","RAM ГБ"),
                                                  ("ram_pct","RAM %"),("net","Сеть"),("disk","Диск")]:
                                    info = pt.get(key, {})
                                    r2   = mtr.get(f"r2_{key}")
                                    mae  = mtr.get(f"mae_{key}")
                                    mape = mtr.get(f"mape_{key}")
                                    rows.append({
                                        "Показатель": lbl,
                                        "Алгоритм":   info.get("model_type", "—"),
                                        "R²":         f"{r2:.3f}"    if r2   is not None else "—",
                                        "MAE":        f"{mae:.2f}"   if mae  is not None else "—",
                                        "MAPE":       f"{mape:.1f}%" if mape is not None else "—",
                                    })
                                st.dataframe(rows, use_container_width=True, hide_index=True)

                if result.get("skipped_servers"):
                    st.warning(f"Серверы без обученной модели (пропущены): {', '.join(result['skipped_servers'])}")


with tab_metrics:
    st.header("Живые метрики из Prometheus")
    st.markdown("Графики в реальном времени напрямую из Prometheus.")

    col_window, col_inst = st.columns([1, 2])
    with col_window:
        window_minutes = st.selectbox(
            "Временное окно", [15, 30, 60, 120, 240], index=2,
            format_func=lambda x: f"Last {x} min",
        )
    with col_inst:
        instance_filter = st.text_input(
            "Фильтр по instance (оставьте пустым для всех)",
            placeholder='e.g. gateway',
            help='Filters node_exporter metrics by instance label',
        )

    inst_selector = f'{{instance="{instance_filter}"}}' if instance_filter else ""

    st.subheader("Бизнес-метрика — Запросы / мин")

    groups = get("/groups/") or []
    if groups:
        bm_group = st.selectbox(
            "Формула группы для отображения",
            options=groups,
            format_func=lambda g: f"{g['name']} → {g['business_metric_name']}",
            key="bm_group",
        )
        bm_formula = bm_group["business_metric_formula"]
    else:
        bm_formula = st.text_input(
            "PromQL выражение",
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
        st.warning("Нет данных из Prometheus — проверьте URL или формулу.")

    st.markdown("---")

    st.subheader("Системные метрики")

    system_queries = [
        ("CPU %",
         f'100 - avg(rate(node_cpu_seconds_total{inst_selector}{{mode="idle"}}[5m])) * 100 by (instance)'
         if instance_filter else
         '100 - avg by (instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100',
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
         f'sum(rate(node_network_receive_bytes_total{inst_selector}[5m])) * 8 / 1048576'
         if instance_filter else
         'sum by (instance)(rate(node_network_receive_bytes_total[5m])) * 8 / 1048576',
         COLOURS["net"], "Mbps", None),

        ("Disk IO %",
         f'avg(rate(node_disk_io_time_seconds_total{inst_selector}[5m])) * 100'
         if instance_filter else
         'avg by (instance)(rate(node_disk_io_time_seconds_total[5m])) * 100',
         COLOURS["disk"], "%", [0, 100]),
    ]

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


with tab_jobs:
    st.header("Задания обучения")

    configs = get("/configs/") or []
    if not configs:
        st.info("Конфигурации не найдены. Создайте группу серверов и создайте конфиги.")
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
            st.subheader("Запустить обучение")
            lookback = st.slider("Дней истории", 1, 90, 7, key="jobs_lookback")
            if st.button("Обучить", type="primary", key="jobs_train_btn"):
                result = post(f"/configs/{cfg_id}/train/", {"lookback_days": lookback})
                if result:
                    st.success(f"Job submitted — id={result['job_id']}  status={result['status']}")

            st.markdown("---")
            st.subheader("История заданий")
            jobs = get(f"/configs/{cfg_id}/train/jobs") or []
            if jobs:
                STATUS_ICON = {"queued": "в очереди", "running": "выполняется", "done": "завершено", "failed": "ошибка"}
                for job in jobs[:10]:
                    icon = STATUS_ICON.get(job["status"], "—")
                    dur  = f"  ({job['duration_seconds']:.0f}s)" if job.get("duration_seconds") else ""
                    st.markdown(f"{icon} Job **{job['id']}** — `{job['status']}`{dur}")
                    if job.get("error_message"):
                        with st.expander("Подробности ошибки"):
                            st.code(job["error_message"])
            else:
                st.caption("Заданий нет.")

        with col_models:
            st.subheader("Обученные модели")
            models = get(f"/configs/{cfg_id}/train/models") or []
            if models:
                for model in models[:5]:
                    status_colour = {"ready": "готова", "training": "обучается",
                                     "failed": "ошибка", "pending": "ожидание"}.get(model["status"], "—")
                    st.markdown(
                        f"{status_colour} **v{model['version']}** — "
                        f"`{model['algorithm']}`  "
                        f"лаг={model.get('lag_minutes', '?')} мин"
                    )
                    if model.get("metrics") and model["status"] == "ready":
                        m = model["metrics"]
                        params = model.get("parameters", {}) or {}
                        pt = params.get("per_target", {}) or {}
                        target_rows = [
                            ("CPU",    "cpu",    COLOURS["cpu"]),
                            ("RAM GB", "ram_gb", COLOURS["ram_gb"]),
                            ("RAM %",  "ram_pct",COLOURS["ram_pct"]),
                            ("Сеть",   "net",    COLOURS["net"]),
                            ("Диск",   "disk",   COLOURS["disk"]),
                        ]
                        hdr = st.columns([2,1,1,1,1,1])
                        hdr[0].caption("Показатель")
                        hdr[1].caption("Алгоритм")
                        hdr[2].caption("R²")
                        hdr[3].caption("MAE")
                        hdr[4].caption("RMSE")
                        hdr[5].caption("MAPE %")
                        for t_label, t_key, t_color in target_rows:
                            r2   = m.get(f"r2_{t_key}")
                            mae  = m.get(f"mae_{t_key}")
                            rmse = m.get(f"rmse_{t_key}")
                            mape = m.get(f"mape_{t_key}")
                            mtype= pt.get(t_key, {}).get("model_type", "?")
                            r2_icon = ""
                            row = st.columns([2,1,1,1,1,1])
                            row[0].markdown(f"<span style='color:{t_color}'>■</span> **{t_label}**", unsafe_allow_html=True)
                            row[1].caption(mtype or "?")
                            row[2].caption(f"{r2_icon} {r2:.3f}" if r2 is not None else "—")
                            row[3].caption(f"{mae:.3f}" if mae is not None else "—")
                            row[4].caption(f"{rmse:.3f}" if rmse is not None else "—")
                            row[5].caption(f"{mape:.1f}" if mape is not None else "—")
                        st.caption(f"MAPE общий: {m.get('mape_overall', '?'):.1f}%  |  шаг={params.get('best_step_seconds','?')}с")
                    st.markdown("---")
            else:
                st.caption("Моделей нет.")


with tab_accuracy:
    st.header("Точность модели и дрейф данных")

    configs = get("/configs/") or []
    if not configs:
        st.info("Конфигурации не найдены.")
    else:
        acc_cfg = st.selectbox(
            "Config",
            options=configs,
            format_func=lambda c: f"{c['name']} (id={c['id']})",
            key="acc_cfg",
        )
        cfg_id = acc_cfg["id"]

        models = get(f"/configs/{cfg_id}/train/models") or []
        ready  = [m for m in models if m["status"] == "ready"]

        if not ready:
            st.warning("Для этой конфигурации нет обученной модели.")
        else:
            model_id = ready[0]["id"]

            col_status, col_eval = st.columns([2, 1])

            with col_status:
                status = get(f"/models/{model_id}/accuracy")
                if status:
                    health_icon = "Норма" if status["is_healthy"] else "Проблема"
                    health_colour = COLOURS["healthy"] if status["is_healthy"] else COLOURS["warning"]
                    st.markdown(
                        f"<h3 style='color:{health_colour}'>{health_icon}</h3>",
                        unsafe_allow_html=True,
                    )
                    if status.get("health_reason"):
                        st.caption(status["health_reason"])

                    st.metric("Оценок выполнено", status["n_evaluations"])
                    st.metric("Оценено точек", status["n_samples_total"])

                    ev = status.get("latest_evaluation")
                    if ev:
                        st.markdown("**Последняя оценка**")
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
                            st.warning("Данная оценка инициировала автоматическое retraining.")

            with col_eval:
                st.markdown("**Действия**")
                if st.button("Принудительная оценка", use_container_width=True):
                    with st.spinner("Выполняется оценка..."):
                        result = post(f"/models/{model_id}/accuracy/evaluate", {})
                    if result:
                        st.success("Оценка завершена.")
                        st.rerun()
                    else:
                        st.error("Ошибка оценки — нужно больше прогнозов с фактическими значениями.")

            st.markdown("---")
            st.subheader("История R²")
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
                    ("r2_net",     "Сеть", COLOURS["net"]),
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
                                 annotation_text="R²=0.85 — порог")
                fig_r2.update_layout(
                    height=300,
                    yaxis={"range": [-0.1, 1.05], "title": "R²"},
                    margin={"t": 10, "b": 30, "l": 50, "r": 10},
                    legend={"orientation": "h", "y": -0.25},
                )
                st.plotly_chart(fig_r2, use_container_width=True)

                if "psi_value" in df_hist.columns:
                    st.subheader("История дрейфа (PSI)")
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
                st.info("Запустите минимум 2 оценки для отображения истории.")
