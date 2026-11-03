"""Executive Streamlit Dashboard for Customer Segmentation & Churn Prevention.

Enterprise intelligence platform featuring RFM feature exploration, K-Means clustering,
Elbow & Silhouette validation, Cohort Retention Curves, and Churn Risk Simulator.
Strictly adheres to executive aesthetics and corporate dark theme.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COHORT_BENCHMARK,
    ELBOW_REFERENCE_DATA,
    PERSONAS_METADATA,
    PROCESSED_RFM_PATH,
)
from src.models.churn import ChurnRiskEstimator
from src.models.kmeans import KMeansSegmentation

# Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Intelligence & Churn Prevention",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS for Unified Modern UI/UX
st.markdown(
    """
    <style>
    /* Global App Container */
    .stApp {
        background-color: #0a0e17;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0a0e17;
    }
    [data-testid="stHeader"] {
        background-color: rgba(10, 14, 23, 0.85);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] {
        background-color: #0e1422;
        border-right: 1px solid #1f2937;
    }

    /* Headings and Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, span, label, div {
        color: #cbd5e1;
    }

    /* Modern KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: linear-gradient(180deg, #131b2e 0%, #0f1626 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
        border-color: #38bdf8;
    }
    .kpi-card-total { border-top: 3px solid #818cf8; }
    .kpi-card-vip { border-top: 3px solid #10b981; }
    .kpi-card-potential { border-top: 3px solid #38bdf8; }
    .kpi-card-risk { border-top: 3px solid #f43f5e; }
    .kpi-card-hibernating { border-top: 3px solid #f59e0b; }

    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
        font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, Consolas, monospace;
        margin-bottom: 6px;
    }
    .kpi-meta {
        font-size: 0.78rem;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Badges */
    .kpi-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-sky { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-rose { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-indigo { background: rgba(129, 140, 248, 0.15); color: #a5b4fc; border: 1px solid rgba(129, 140, 248, 0.3); }

    /* Modern Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 4px;
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 9px 18px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* Action Cards & Info Boxes */
    .panel-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    .panel-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8b949e;
        margin-bottom: 6px;
    }

    /* Form Controls Dark Override */
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border-color: #1f2937 !important;
        color: #f8fafc !important;
    }
    .stSlider > div [data-baseweb="slider"] {
        color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_corporate_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 440,
    show_legend: bool = True,
) -> go.Figure:
    """Standardizes Plotly figures with high-contrast corporate dark aesthetics."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0a0e17",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            color="#e2e8f0",
            size=12,
        ),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#f8fafc"),
            x=0.01,
            y=0.96,
        ),
        margin=dict(l=48, r=32, t=54 if title else 24, b=48),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17, 24, 39, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="monospace",
            font_color="#f8fafc",
            bordercolor="#38bdf8",
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="#1f2937",
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="#1f2937",
    )
    return fig


# Header Section
st.markdown(
    """
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #1f2937; margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span class="kpi-badge badge-sky">ENTERPRISE ML PLATFORM</span>
            <span class="kpi-badge badge-emerald">K-MEANS k=4 (SILHOUETTE 0.68)</span>
            <span class="kpi-badge badge-indigo">BASE ATIVA: 3.000 CLIENTES</span>
        </div>
        <h1 style="color: #f8fafc; margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;">
            Segmentação de Clientes & Prevenção de Churn
        </h1>
        <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 0.96rem; max-width: 950px;">
            Plataforma preditiva corporativa: análise matricial RFM, clusterização não-supervisionada,
            mapeamento de personas acionáveis e scoring contínuo de propensão à evasão de receita.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads clustered RFM dataset or generates synthetic benchmark."""
    if PROCESSED_RFM_PATH.exists():
        return pd.read_parquet(PROCESSED_RFM_PATH)

    from src.features.rfm import RFMEngine

    eng = RFMEngine()
    tx_df, _ = eng.generate_synthetic_transactions(n_customers=3000, random_state=42)
    rfm_df = eng.calculate_rfm(tx_df)
    km = KMeansSegmentation(n_clusters=4, random_state=42)
    return km.fit(rfm_df)


@st.cache_resource
def get_fitted_kmeans(df: pd.DataFrame) -> KMeansSegmentation:
    """Caches fitted KMeans pipeline to avoid re-fitting during user interactions."""
    km_engine = KMeansSegmentation(n_clusters=4, random_state=42)
    km_engine.fit(df)
    return km_engine


df_rfm = load_data()
km_model = get_fitted_kmeans(df_rfm)

# Compute Top Level Metrics
n_total = len(df_rfm)
total_gmv = df_rfm["monetary"].sum()
vips_cnt = len(df_rfm[df_rfm["persona_key"] == "vip"])
pot_cnt = len(df_rfm[df_rfm["persona_key"] == "potential"])
risk_cnt = len(df_rfm[df_rfm["persona_key"] == "at_risk"])
hib_cnt = len(df_rfm[df_rfm["persona_key"] == "hibernating"])

# 5 Modern Top KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-total">
            <div class="kpi-title">
                <span>Base Total</span>
                <span class="kpi-badge badge-indigo">100%</span>
            </div>
            <div class="kpi-value">{n_total:,}</div>
            <div class="kpi-meta">GMV Total: <b>R$ {total_gmv/1e6:.2f}M</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-vip">
            <div class="kpi-title">
                <span>Champions VIPs</span>
                <span class="kpi-badge badge-emerald">{(vips_cnt/n_total)*100:.1f}%</span>
            </div>
            <div class="kpi-value" style="color: #34d399;">{vips_cnt:,}</div>
            <div class="kpi-meta">Churn Base: <b>3.2%</b> | Alto LTV</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-potential">
            <div class="kpi-title">
                <span>Potenciais Leais</span>
                <span class="kpi-badge badge-sky">{(pot_cnt/n_total)*100:.1f}%</span>
            </div>
            <div class="kpi-value" style="color: #38bdf8;">{pot_cnt:,}</div>
            <div class="kpi-meta">Churn Base: <b>12.5%</b> | Expansão</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-risk">
            <div class="kpi-title">
                <span>Em Risco Alto</span>
                <span class="kpi-badge badge-rose">{(risk_cnt/n_total)*100:.1f}%</span>
            </div>
            <div class="kpi-value" style="color: #fb7185;">{risk_cnt:,}</div>
            <div class="kpi-meta">Churn Base: <b>68.4%</b> | Alerta CS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-hibernating">
            <div class="kpi-title">
                <span>Hibernando</span>
                <span class="kpi-badge badge-amber">{(hib_cnt/n_total)*100:.1f}%</span>
            </div>
            <div class="kpi-value" style="color: #fbbf24;">{hib_cnt:,}</div>
            <div class="kpi-meta">Churn Base: <b>89.0%</b> | Reativar</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Navigation Tabs
tab_clusters, tab_elbow, tab_cohort, tab_sim, tab_explorer = st.tabs([
    "Clusterização K-Means & RFM",
    "Método do Cotovelo & Validação",
    "Retenção de Cohort (M0-M6)",
    "Simulador de Churn & Diagnóstico",
    "Explorador de Clientes & Exportação",
])

# Color mapping consistent across the entire platform
COLOR_MAP = {
    "Champions / Clientes VIPs": "#10b981",
    "Potenciais Leais": "#38bdf8",
    "Em Risco Alto (Ticket Alto)": "#f43f5e",
    "Hibernando / Baixo Engajamento": "#f59e0b",
}


# ==========================================
# TAB 1: CLUSTERS & RFM
# ==========================================
with tab_clusters:
    st.markdown("### Mapeamento Multidimensional de Clusters RFM")
    st.markdown(
        "A segmentação particiona a base de clientes a partir das dimensões normalizadas de "
        "**Recência** (tempo desde a última compra), **Frequência** (volume de pedidos) e "
        "**Monetário** (receita total acumulada)."
    )

    # Interactive Controls Bar
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        selected_personas = st.multiselect(
            "Filtrar Personas no Gráfico:",
            options=list(COLOR_MAP.keys()),
            default=list(COLOR_MAP.keys()),
        )
    with ctrl_col2:
        view_mode = st.selectbox(
            "Modo de Visualização:",
            [
                "2D: Frequência vs Faturamento (R$)",
                "2D: Recência (dias) vs Frequência",
                "3D: Espaço Tridimensional RFM",
            ],
        )
    with ctrl_col3:
        use_log_y = st.toggle("Escala Logarítmica (Monetário)", value=False)

    # Filter dataframe based on persona selection
    df_filtered = df_rfm[df_rfm["persona_name"].isin(selected_personas)]

    if df_filtered.empty:
        st.warning("Selecione ao menos uma persona para visualizar os dados.")
    else:
        if view_mode == "2D: Frequência vs Faturamento (R$)":
            fig_scatter = px.scatter(
                df_filtered,
                x="frequency",
                y="monetary",
                color="persona_name",
                color_discrete_map=COLOR_MAP,
                size="recency",
                size_max=16,
                custom_data=["customer_id", "persona_name", "recency", "rfm_score"],
                labels={
                    "frequency": "Frequência (Total de Compras)",
                    "monetary": "Faturamento Total (R$)",
                    "persona_name": "Segmento",
                    "recency": "Recência (Dias)",
                },
                log_y=use_log_y,
            )
            fig_scatter.update_traces(
                hovertemplate=(
                    "<b>ID:</b> %{customdata[0]}<br>"
                    "<b>Segmento:</b> %{customdata[1]}<br>"
                    "<b>Frequência:</b> %{x} compras<br>"
                    "<b>Monetário:</b> R$ %{y:,.2f}<br>"
                    "<b>Recência:</b> %{customdata[2]} dias<br>"
                    "<b>Score RFM:</b> %{customdata[3]}<extra></extra>"
                )
            )
            apply_corporate_layout(
                fig_scatter,
                title=f"Dispersão de Comportamento: Frequência x Receita ({len(df_filtered):,} clientes)",
                height=460,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        elif view_mode == "2D: Recência (dias) vs Frequência":
            fig_scatter = px.scatter(
                df_filtered,
                x="recency",
                y="frequency",
                color="persona_name",
                color_discrete_map=COLOR_MAP,
                size="monetary",
                size_max=18,
                custom_data=["customer_id", "persona_name", "monetary", "rfm_score"],
                labels={
                    "recency": "Recência (Dias Sem Comprar)",
                    "frequency": "Frequência (Total de Compras)",
                    "persona_name": "Segmento",
                },
            )
            fig_scatter.update_traces(
                hovertemplate=(
                    "<b>ID:</b> %{customdata[0]}<br>"
                    "<b>Segmento:</b> %{customdata[1]}<br>"
                    "<b>Recência:</b> %{x} dias<br>"
                    "<b>Frequência:</b> %{y} compras<br>"
                    "<b>Monetário:</b> R$ %{customdata[2]:,.2f}<extra></extra>"
                )
            )
            apply_corporate_layout(
                fig_scatter,
                title=f"Matriz de Inatividade vs Engajamento: Recência x Frequência ({len(df_filtered):,} clientes)",
                height=460,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            # 3D RFM Space
            fig_3d = px.scatter_3d(
                df_filtered,
                x="recency",
                y="frequency",
                z="monetary",
                color="persona_name",
                color_discrete_map=COLOR_MAP,
                opacity=0.85,
                custom_data=["customer_id", "persona_name", "rfm_score"],
                labels={
                    "recency": "Recência (Dias)",
                    "frequency": "Frequência",
                    "monetary": "Monetário (R$)",
                    "persona_name": "Segmento",
                },
            )
            fig_3d.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111827",
                height=520,
                scene=dict(
                    xaxis=dict(backgroundcolor="#0a0e17", gridcolor="rgba(255,255,255,0.08)", title="Recência (dias)"),
                    yaxis=dict(backgroundcolor="#0a0e17", gridcolor="rgba(255,255,255,0.08)", title="Frequência"),
                    zaxis=dict(backgroundcolor="#0a0e17", gridcolor="rgba(255,255,255,0.08)", title="Monetário (R$)"),
                ),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    # Strategic Action Playbooks Matrix
    st.markdown("#### Matriz Estratégica de Personas & Playbooks de Ação CRM")
    persona_summary_list = []
    for p_key, meta in PERSONAS_METADATA.items():
        sub = df_rfm[df_rfm["persona_key"] == p_key]
        cnt = len(sub)
        persona_summary_list.append({
            "Persona": meta["name"],
            "Participação": (cnt / n_total),
            "Clientes": cnt,
            "Recência Média": f"{sub['recency'].mean():.0f} dias",
            "Frequência Média": f"{sub['frequency'].mean():.1f} compras",
            "Ticket Médio Total": sub["monetary"].mean(),
            "Risco Churn": meta["base_churn_rate"],
            "Plano de Ação Recomendado": meta["action_playbook"],
        })

    df_persona_table = pd.DataFrame(persona_summary_list)
    st.dataframe(
        df_persona_table,
        column_config={
            "Participação": st.column_config.ProgressColumn(
                "Participação", format="%.1f%%", min_value=0.0, max_value=1.0
            ),
            "Clientes": st.column_config.NumberColumn("Qtd. Clientes", format="%d"),
            "Ticket Médio Total": st.column_config.NumberColumn("Ticket Médio", format="R$ %.2f"),
            "Risco Churn": st.column_config.ProgressColumn(
                "Risco Base Churn", format="%.1f%%", min_value=0.0, max_value=1.0
            ),
            "Plano de Ação Recomendado": st.column_config.TextColumn(
                "Diretriz Tática de CRM", width="large"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )


# ==========================================
# TAB 2: ELBOW & VALIDATION
# ==========================================
with tab_elbow:
    st.markdown("### Validação Estatística do Número Ótimo de Clusters (k=4)")
    st.markdown(
        "A escolha de **k=4** foi fundamentada pela análise de trade-off entre perda de informação "
        "e interpretabilidade prática de negócio, garantindo a maior separabilidade média de silhueta."
    )

    ev_col1, ev_col2, ev_col3 = st.columns(3)
    with ev_col1:
        st.markdown(
            """
            <div class="panel-card" style="border-left: 4px solid #38bdf8;">
                <div class="panel-title">Ponto de Inflexão (Cotovelo)</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8; font-family: monospace;">k = 4</div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                    Queda de 70.8% no WCSS em relação a k=2, marcando a saturação da inércia.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ev_col2:
        st.markdown(
            """
            <div class="panel-card" style="border-left: 4px solid #10b981;">
                <div class="panel-title">Pico do Silhouette Score</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #10b981; font-family: monospace;">0.68</div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                    Excelente coesão interna intra-cluster e nítida separação das 4 personas.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ev_col3:
        st.markdown(
            """
            <div class="panel-card" style="border-left: 4px solid #818cf8;">
                <div class="panel-title">Interpretabilidade de Negócio</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #818cf8; font-family: monospace;">4 Personas</div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                    Perfis operacionais claros para os times de Growth, CRM e Customer Success.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    df_elbow = pd.DataFrame(ELBOW_REFERENCE_DATA)

    fig_elbow = go.Figure()
    # Inertia line
    fig_elbow.add_trace(go.Scatter(
        x=df_elbow["k"],
        y=df_elbow["inertia"],
        mode="lines+markers",
        name="Inércia WCSS (Cotovelo)",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=9, color="#38bdf8", symbol="circle"),
        yaxis="y1",
    ))
    # Silhouette line
    fig_elbow.add_trace(go.Scatter(
        x=df_elbow["k"],
        y=df_elbow["silhouette"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#10b981", width=3, dash="dash"),
        marker=dict(size=9, color="#10b981", symbol="diamond"),
        yaxis="y2",
    ))
    # Optimal vertical line
    fig_elbow.add_vline(
        x=4,
        line_dash="dot",
        line_color="#fbbf24",
        line_width=2,
        annotation_text="Ótimo Global (k = 4 | Silhouette = 0.68)",
        annotation_position="top right",
        annotation_font=dict(color="#fbbf24", size=11),
    )

    fig_elbow.update_layout(
        title="Curva de Validação: Inércia Intra-Cluster (WCSS) vs Silhouette Score",
        xaxis=dict(title="Número de Clusters (k)", tickmode="linear", tick0=2, dtick=1),
        yaxis=dict(title="Inércia (WCSS)", title_font=dict(color="#38bdf8")),
        yaxis2=dict(
            title="Silhouette Score",
            title_font=dict(color="#10b981"),
            overlaying="y",
            side="right",
            range=[0.3, 0.8],
        ),
    )
    apply_corporate_layout(fig_elbow, height=420)
    st.plotly_chart(fig_elbow, use_container_width=True)


# ==========================================
# TAB 3: COHORT RETENTION
# ==========================================
with tab_cohort:
    st.markdown("### Curvas de Sobrevivência e Retenção de Cohort (M0 a M6)")
    st.markdown(
        "Acompanhamento da taxa de retenção longitudinal ao longo de 6 meses pós-aquisição, "
        "revelando o contraste entre a fidelização resiliente dos **VIPs (94% em M6)** e a rápida "
        "evasão dos clientes **Hibernando (9% em M6)**."
    )

    df_cohort = pd.DataFrame(COHORT_BENCHMARK)

    fig_cohort = go.Figure()
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["vip"],
        mode="lines+markers",
        name="Champions VIPs",
        line=dict(color="#10b981", width=3.5),
        marker=dict(size=8),
    ))
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["potential"],
        mode="lines+markers",
        name="Potenciais Leais",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=7),
    ))
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["at_risk"],
        mode="lines+markers",
        name="Em Risco Alto",
        line=dict(color="#f43f5e", width=3),
        marker=dict(size=7),
    ))
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["hibernating"],
        mode="lines+markers",
        name="Hibernando",
        line=dict(color="#f59e0b", width=3, dash="dot"),
        marker=dict(size=7),
    ))

    fig_cohort.update_layout(
        title="Sobrevivência da Base por Persona (M0 a M6)",
        xaxis_title="Mês do Cohort",
        yaxis_title="Retenção Acumulada (%)",
        yaxis=dict(range=[0, 105]),
    )
    apply_corporate_layout(fig_cohort, height=410)
    st.plotly_chart(fig_cohort, use_container_width=True)

    # Detailed Table
    st.markdown("#### Matriz Numérica de Retenção por Mês (%)")
    st.dataframe(
        df_cohort.rename(columns={
            "month_label": "Mês do Cohort",
            "vip": "Champions VIPs (%)",
            "potential": "Potenciais Leais (%)",
            "at_risk": "Em Risco Alto (%)",
            "hibernating": "Hibernando (%)",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ==========================================
# TAB 4: CHURN SIMULATOR & WHAT-IF ANALYSIS
# ==========================================
with tab_sim:
    st.markdown("### Simulador de Risco de Churn & Classificador em Tempo Real")
    st.markdown(
        "Ajuste os parâmetros transacionais ou selecione um perfil pré-configurado para prever "
        "a probabilidade de evasão, o segmento atribuído e o plano de ação de retenção correspondente."
    )

    # Preset Quick Buttons
    st.markdown("##### Perfis Típicos Rápidos (Clique para simular):")
    p_b1, p_b2, p_b3, p_b4 = st.columns(4)
    if "sim_rec" not in st.session_state:
        st.session_state.sim_rec = 25
    if "sim_freq" not in st.session_state:
        st.session_state.sim_freq = 8
    if "sim_mon" not in st.session_state:
        st.session_state.sim_mon = 5200.0

    with p_b1:
        if st.button("Perfil VIP (Alto Valor)", use_container_width=True):
            st.session_state.sim_rec = 4
            st.session_state.sim_freq = 22
            st.session_state.sim_mon = 10500.0
            st.rerun()
    with p_b2:
        if st.button("Perfil Potencial Leal", use_container_width=True):
            st.session_state.sim_rec = 18
            st.session_state.sim_freq = 10
            st.session_state.sim_mon = 5500.0
            st.rerun()
    with p_b3:
        if st.button("Perfil Em Risco Alto", use_container_width=True):
            st.session_state.sim_rec = 78
            st.session_state.sim_freq = 5
            st.session_state.sim_mon = 6800.0
            st.rerun()
    with p_b4:
        if st.button("Perfil Hibernando", use_container_width=True):
            st.session_state.sim_rec = 142
            st.session_state.sim_freq = 2
            st.session_state.sim_mon = 1200.0
            st.rerun()

    sim_col1, sim_col2 = st.columns([1, 1.4])
    with sim_col1:
        st.markdown("#### Parâmetros do Cliente")
        input_rec = st.slider(
            "Recência: Dias Sem Comprar",
            min_value=1,
            max_value=240,
            value=st.session_state.sim_rec,
            step=1,
            help="Dias decorridos desde a última transação do cliente.",
        )
        input_freq = st.slider(
            "Frequência: Total de Pedidos",
            min_value=1,
            max_value=40,
            value=st.session_state.sim_freq,
            step=1,
            help="Número total histórico de compras realizadas.",
        )
        input_mon = st.number_input(
            "Faturamento Total Gasto (R$)",
            min_value=100.0,
            max_value=50000.0,
            value=float(st.session_state.sim_mon),
            step=250.0,
            help="Receita bruta monetária gerada pelo cliente.",
        )

        # Inference
        pred = km_model.predict_single(recency=input_rec, frequency=input_freq, monetary=input_mon)
        churn_prof = ChurnRiskEstimator.estimate_churn(
            recency_days=float(input_rec),
            frequency=float(input_freq),
            monetary=float(input_mon),
            persona_key=pred.persona_key,
        )

        # Status badge determination
        if churn_prof.churn_probability < 0.25:
            risk_color = "#10b981"
            badge_class = "badge-emerald"
        elif churn_prof.churn_probability < 0.60:
            risk_color = "#f59e0b"
            badge_class = "badge-amber"
        else:
            risk_color = "#f43f5e"
            badge_class = "badge-rose"

        st.markdown(
            f"""
            <div class="panel-card" style="margin-top: 14px; border-left: 4px solid {risk_color};">
                <div class="panel-title">Persona Classificada</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                    {pred.persona_name}
                </div>
                <div style="font-size: 0.80rem; color: #94a3b8;">
                    Distância euclidiana ao centroide: <b>{pred.distance_to_centroid:.4f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sim_col2:
        st.markdown("#### Diagnóstico Executivo de Churn")

        # Speedometer Gauge Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=churn_prof.churn_probability * 100,
                domain=dict(x=[0, 1], y=[0, 1]),
                number=dict(suffix="%", font=dict(size=36, color=risk_color, family="monospace")),
                title=dict(
                    text="<b>Probabilidade Preditiva de Churn</b>",
                    font=dict(size=14, color="#e2e8f0"),
                ),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94a3b8"),
                    bar=dict(color=risk_color, thickness=0.25),
                    bgcolor="#1e293b",
                    borderwidth=1,
                    bordercolor="#334155",
                    steps=[
                        dict(range=[0, 25], color="rgba(16, 185, 129, 0.20)"),
                        dict(range=[25, 60], color="rgba(245, 158, 11, 0.20)"),
                        dict(range=[60, 100], color="rgba(244, 63, 94, 0.20)"),
                    ],
                    threshold=dict(
                        line=dict(color="#f8fafc", width=3),
                        thickness=0.8,
                        value=churn_prof.churn_probability * 100,
                    ),
                ),
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="#111827",
            font=dict(color="#f8fafc"),
            height=230,
            margin=dict(l=30, r=30, t=40, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Action Recommendations
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="panel-title" style="margin: 0;">Classificação de Risco</span>
                    <span class="kpi-badge {badge_class}">{churn_prof.risk_tier}</span>
                </div>
                <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 10px;">
                    <b>Principal Fator:</b> {churn_prof.primary_risk_driver}
                </div>
                <div class="panel-title" style="margin-top: 10px;">Plano Tático Recomendado (CRM):</div>
                <div style="color: #38bdf8; font-size: 0.92rem; line-height: 1.4;">
                    {churn_prof.action_plan}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Survival Decay Curve for this customer
        months_proj = list(churn_prof.retention_projection.keys())
        ret_vals = list(churn_prof.retention_projection.values())

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=months_proj,
            y=ret_vals,
            mode="lines+markers",
            name="Curva de Retenção",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=7, color="#38bdf8"),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
        ))
        fig_proj.update_layout(
            title="Projeção Mensal de Sobrevivência deste Cliente (M0 a M6)",
            xaxis_title="Mês do Ciclo",
            yaxis_title="Probabilidade de Retenção (%)",
            yaxis=dict(range=[0, 105]),
        )
        apply_corporate_layout(fig_proj, height=240, show_legend=False)
        st.plotly_chart(fig_proj, use_container_width=True)


# ==========================================
# TAB 5: DATA EXPLORER & CSV EXPORT
# ==========================================
with tab_explorer:
    st.markdown("### Explorador de Base de Clientes Segmentada")
    st.markdown(
        "Pesquise clientes individuais, filtre por persona e exporte a base consolidada para ativação em ferramentas de CRM."
    )

    exp_col1, exp_col2, exp_col3 = st.columns([1.5, 1.5, 1])
    with exp_col1:
        search_id = st.text_input("Buscar por ID do Cliente (ex: CUST-00042):", "")
    with exp_col2:
        exp_persona = st.selectbox("Filtrar por Segmento:", ["Todos os Segmentos"] + list(COLOR_MAP.keys()))
    with exp_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        csv_data = df_rfm.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar Base (CSV)",
            data=csv_data,
            file_name="clientes_segmentados_rfm.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Filtered dataframe
    df_display = df_rfm.copy()
    if search_id.strip():
        df_display = df_display[df_display["customer_id"].str.contains(search_id.strip(), case=False)]
    if exp_persona != "Todos os Segmentos":
        df_display = df_display[df_display["persona_name"] == exp_persona]

    st.dataframe(
        df_display[[
            "customer_id",
            "persona_name",
            "recency",
            "frequency",
            "monetary",
            "avg_ticket",
            "rfm_score",
            "action_playbook",
        ]],
        column_config={
            "customer_id": "ID Cliente",
            "persona_name": "Segmento",
            "recency": st.column_config.NumberColumn("Recência (dias)", format="%d"),
            "frequency": st.column_config.NumberColumn("Frequência", format="%d"),
            "monetary": st.column_config.NumberColumn("Valor Total", format="R$ %.2f"),
            "avg_ticket": st.column_config.NumberColumn("Ticket Médio", format="R$ %.2f"),
            "rfm_score": "Score RFM",
            "action_playbook": st.column_config.TextColumn("Plano CRM", width="large"),
        },
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Mostrando {len(df_display):,} de {n_total:,} clientes registrados.")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1f2937; margin-bottom: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.08em; text-transform: uppercase;">
                Customer Intelligence
            </div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">
                Gestão Preditiva de Churn
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Resumo da Arquitetura")
    st.markdown(
        """
        - **Pipeline RFM**: Ingestão transacional e normalização logarítmica.
        - **Algoritmo**: K-Means Não-Supervisionado ($k=4$).
        - **Validação**: Silhouette Score **0.68** e Método do Cotovelo.
        - **Decisão**: Playbooks operacionais de CRM para cada cohort.
        """
    )

    st.markdown("---")
    st.markdown("#### Documentação da API REST")
    st.markdown(
        """
        Para integrações em tempo real via backend, acesse os endpoints documentados no Swagger UI:
        - **Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
        - **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
        """
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.80rem; color: #8b949e; line-height: 1.4;">
            <b>Desenvolvido por:</b> Renan Nocelli<br>
            Engenharia de Dados & Machine Learning<br>
            <a href="https://renan-nocelli.vercel.app" target="_blank" style="color: #38bdf8; text-decoration: none;">
                renan-nocelli.vercel.app
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
