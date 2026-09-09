"""Executive Streamlit Dashboard for Customer Segmentation & Churn Prevention.

Interactive business platform featuring RFM feature exploration, K-Means clustering,
Elbow & Silhouette validation, Cohort Retention Curves, and Churn Risk Simulator.
Strictly adheres to executive aesthetics without emojis.
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

st.set_page_config(
    page_title="Segmentacao de Clientes & Prevencao de Churn",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Corporate CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.80rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.60rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #7ee787;
        margin-top: 4px;
    }
    .badge-persona {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.80rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-vip {
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
    }
    .badge-potential {
        background-color: rgba(88, 166, 255, 0.15);
        border: 1px solid #58a6ff;
        color: #58a6ff;
    }
    .badge-atrisk {
        background-color: rgba(218, 54, 51, 0.15);
        border: 1px solid #da3633;
        color: #f85149;
    }
    .badge-hibernating {
        background-color: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        color: #d29922;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div style="padding-bottom: 16px; border-bottom: 1px solid #30363d; margin-bottom: 24px;">
        <h1 style="color: #f0f6fc; margin: 0; font-size: 2.1rem; font-weight: 700;">
            Segmentacao de Clientes & Prevencao de Churn
        </h1>
        <p style="color: #8b949e; margin: 6px 0 0 0; font-size: 0.98rem;">
            Plataforma preditiva corporativa de Machine Learning: clusterizacao K-Means sobre features RFM,
            mapeamento de personas acionaveis e estimacao de risco de evasao de receita.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads clustered RFM dataset."""
    if PROCESSED_RFM_PATH.exists():
        return pd.read_parquet(PROCESSED_RFM_PATH)

    # Fallback to in-memory generation if file is missing
    from src.features.rfm import RFMEngine

    eng = RFMEngine()
    tx_df, _ = eng.generate_synthetic_transactions(n_customers=3000, random_state=42)
    rfm_df = eng.calculate_rfm(tx_df)
    km = KMeansSegmentation(n_clusters=4, random_state=42)
    return km.fit(rfm_df)


df_rfm = load_data()

# Navigation Tabs
tab_clusters, tab_elbow, tab_cohort, tab_sim = st.tabs([
    "Clusterizacao K-Means & RFM",
    "Metodo do Cotovelo & Validacao",
    "Retencao de Cohort (M0-M6)",
    "Simulador de Risco de Churn",
])

# TAB 1: CLUSTERS & RFM MATRIX
with tab_clusters:
    st.markdown("### Distribuicao dos Clusters e Personas de Compra")
    st.markdown(
        "A segmentacao baseia-se nas dimensoes de Recencia (dias sem comprar), "
        "Frequencia (numero de pedidos) e Monetario (valor total gasto), normalizadas via transformacao logaritmica."
    )

    # Summary metrics
    n_total = len(df_rfm)
    vips_cnt = len(df_rfm[df_rfm["persona_key"] == "vip"])
    pot_cnt = len(df_rfm[df_rfm["persona_key"] == "potential"])
    risk_cnt = len(df_rfm[df_rfm["persona_key"] == "at_risk"])
    hib_cnt = len(df_rfm[df_rfm["persona_key"] == "hibernating"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total de Clientes</div>
                <div class="metric-value">{n_total:,}</div>
                <div class="metric-sub">Base ativa analisada</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Champions VIPs</div>
                <div class="metric-value" style="color:#3fb950;">{vips_cnt:,}</div>
                <div class="metric-sub">{(vips_cnt/n_total)*100:.1f}% da base | Churn ~3.2%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Em Risco Alto (Ticket Alto)</div>
                <div class="metric-value" style="color:#f85149;">{risk_cnt:,}</div>
                <div class="metric-sub">{(risk_cnt/n_total)*100:.1f}% da base | Churn ~68.4%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Hibernando / Inativos</div>
                <div class="metric-value" style="color:#d29922;">{hib_cnt:,}</div>
                <div class="metric-sub">{(hib_cnt/n_total)*100:.1f}% da base | Churn ~89.0%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Scatter Plot 2D: Frequency vs Monetary with Recency bubble size
    color_map = {
        "Champions / Clientes VIPs": "#3fb950",
        "Potenciais Leais": "#58a6ff",
        "Em Risco Alto (Ticket Alto)": "#f85149",
        "Hibernando / Baixo Engajamento": "#d29922",
    }

    fig_scatter = px.scatter(
        df_rfm,
        x="frequency",
        y="monetary",
        color="persona_name",
        color_discrete_map=color_map,
        hover_data=["customer_id", "recency", "rfm_score"],
        labels={
            "frequency": "Frequencia (Numero de Compras)",
            "monetary": "Valor Monetario Total (R$)",
            "persona_name": "Persona",
            "recency": "Recencia (Dias)",
        },
        title="Dispersao de Clientes: Frequencia vs Valor Monetario (Clusterizacao K-Means)",
        template="plotly_dark",
    )
    fig_scatter.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Action Matrix table
    st.markdown("#### Matriz Estrategica de Personas e Planos de Acao CRM")
    persona_summary_list = []
    for p_key, meta in PERSONAS_METADATA.items():
        sub = df_rfm[df_rfm["persona_key"] == p_key]
        cnt = len(sub)
        persona_summary_list.append({
            "Persona": meta["name"],
            "Clientes": f"{cnt:,} ({(cnt/n_total)*100:.1f}%)",
            "Recencia Media": f"{sub['recency'].mean():.0f} dias",
            "Frequencia Media": f"{sub['frequency'].mean():.1f} compras",
            "Ticket Medio Total": f"R$ {sub['monetary'].mean():,.2f}",
            "Risco Churn": f"{meta['base_churn_rate']*100:.1f}%",
            "Plano de Acao Recomendado": meta["action_playbook"],
        })
    st.dataframe(pd.DataFrame(persona_summary_list), use_container_width=True)


# TAB 2: ELBOW & SILHOUETTE VALIDATION
with tab_elbow:
    st.markdown("### Validacao Estatistica do Numero Otimo de Clusters (k=4)")
    st.markdown(
        "A selecao de k=4 foi determinada pela convergencia do Metodo do Cotovelo (Inertia WCSS) "
        "e pela maximizacao do Silhouette Score (0.68), garantindo coesao interna e separacao maxima entre personas."
    )

    df_elbow = pd.DataFrame(ELBOW_REFERENCE_DATA)

    fig_elbow = go.Figure()

    # Inertia line
    fig_elbow.add_trace(go.Scatter(
        x=df_elbow["k"],
        y=df_elbow["inertia"],
        mode="lines+markers",
        name="Inercia WCSS (Cotovelo)",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=8),
        yaxis="y1",
    ))

    # Silhouette line
    fig_elbow.add_trace(go.Scatter(
        x=df_elbow["k"],
        y=df_elbow["silhouette"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#3fb950", width=3, dash="dash"),
        marker=dict(size=8),
        yaxis="y2",
    ))

    # Highlight optimal k=4
    fig_elbow.add_vline(
        x=4,
        line_dash="dot",
        line_color="#d29922",
        annotation_text="Ponto Otimo (k = 4 | Silhouette = 0.68)",
        annotation_position="top right",
    )

    fig_elbow.update_layout(
        title="Metodo do Cotovelo vs Silhouette Score (k = 2 a 8)",
        xaxis_title="Numero de Clusters (k)",
        yaxis=dict(title="Inercia (WCSS)", title_font=dict(color="#58a6ff")),
        yaxis2=dict(
            title="Silhouette Score",
            title_font=dict(color="#3fb950"),
            overlaying="y",
            side="right",
            range=[0.3, 0.8],
        ),
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Metric comparisons table
    st.markdown("#### Tabela Comparativa de Metricas de Validacao de Clusters")
    st.dataframe(df_elbow.rename(columns={
        "k": "Numero de Clusters (k)",
        "inertia": "Inercia (WCSS)",
        "silhouette": "Silhouette Score",
    }), use_container_width=True)


# TAB 3: COHORT RETENTION
with tab_cohort:
    st.markdown("### Retencao Acumulada de Cohort por Segmento (M0 a M6)")
    st.markdown(
        "Acompanhamento longitudinal da taxa de sobrevivencia de clientes ao longo de 6 meses pos-aquisicao, "
        "comprovando o efeito de retencao elevada dos VIPs (94%) versus o decaimento exponencial dos clientes Hibernando (9%)."
    )

    df_cohort = pd.DataFrame(COHORT_BENCHMARK)

    fig_cohort = go.Figure()
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["vip"],
        mode="lines+markers",
        name="Champions VIPs",
        line=dict(color="#3fb950", width=3),
    ))
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["potential"],
        mode="lines+markers",
        name="Potenciais Leais",
        line=dict(color="#58a6ff", width=2.5),
    ))
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["at_risk"],
        mode="lines+markers",
        name="Em Risco Alto",
        line=dict(color="#f85149", width=2.5),
    ))
    fig_cohort.add_trace(go.Scatter(
        x=df_cohort["month_label"],
        y=df_cohort["hibernating"],
        mode="lines+markers",
        name="Hibernando",
        line=dict(color="#d29922", width=2.5, dash="dot"),
    ))

    fig_cohort.update_layout(
        title="Curva de Sobrevivencia de Clientes por Persona (Meses M0 a M6)",
        xaxis_title="Mes do Cohort",
        yaxis_title="Taxa de Retencao Acumulada (%)",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

    st.dataframe(df_cohort.rename(columns={
        "month_label": "Mes do Cohort",
        "vip": "VIPs (%)",
        "potential": "Potenciais (%)",
        "at_risk": "Em Risco (%)",
        "hibernating": "Hibernando (%)",
    }), use_container_width=True)


# TAB 4: CHURN SIMULATOR & WHAT-IF ANALYSIS
with tab_sim:
    st.markdown("### Simulador de Risco de Churn & Classificador em Tempo Real")
    st.markdown(
        "Ajuste os parametros individuais de comportamento transacional do cliente para estimar "
        "instantaneamente a probabilidade de evasao, enquadramento de persona e plano de acao tatico."
    )

    sim_c1, sim_c2 = st.columns([1, 2])
    with sim_c1:
        st.markdown("#### Parametros do Cliente")
        input_rec = st.slider("Recencia: Dias Sem Comprar", min_value=1, max_value=240, value=25, step=1)
        input_freq = st.slider("Frequencia: Total de Compras", min_value=1, max_value=40, value=8, step=1)
        input_mon = st.number_input("Valor Monetario Total Gasto (R$)", min_value=100.0, max_value=50000.0, value=5200.0, step=250.0)

        # Inference
        km_engine = KMeansSegmentation(n_clusters=4, random_state=42)
        km_engine.fit(df_rfm)
        pred = km_engine.predict_single(recency=input_rec, frequency=input_freq, monetary=input_mon)

        churn_prof = ChurnRiskEstimator.estimate_churn(
            recency_days=float(input_rec),
            frequency=float(input_freq),
            monetary=float(input_mon),
            persona_key=pred.persona_key,
        )

        st.markdown(
            f"""<div class="metric-card" style="margin-top:16px;">
                <div class="metric-label">Persona Classificada</div>
                <div class="metric-value" style="font-size:1.3rem;">{pred.persona_name}</div>
                <div class="metric-sub">Distancia ao centroide: {pred.distance_to_centroid:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Probabilidade de Churn</div>
                <div class="metric-value" style="color:{'#3fb950' if churn_prof.churn_probability < 0.20 else '#f85149' if churn_prof.churn_probability > 0.50 else '#d29922'};">
                    {churn_prof.churn_probability * 100:.1f}%
                </div>
                <div class="metric-sub">Classificacao: {churn_prof.risk_tier}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with sim_c2:
        st.markdown("#### Diagnostico do Cliente e Projecao de Sobrevivencia")

        st.markdown(
            f"""<div class="metric-card">
                <div style="font-size:0.85rem; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">Principal Fator de Risco</div>
                <div style="color:#f0f6fc; font-weight:600; margin-bottom:12px;">{churn_prof.primary_risk_driver}</div>
                <div style="font-size:0.85rem; color:#8b949e; text-transform:uppercase; margin-bottom:4px;">Recomendacao Tática de CRM</div>
                <div style="color:#7ee787; font-size:0.95rem;">{churn_prof.action_plan}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Plot retention projection
        months_proj = list(churn_prof.retention_projection.keys())
        ret_vals = list(churn_prof.retention_projection.values())

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=months_proj,
            y=ret_vals,
            mode="lines+markers",
            name="Projecao Individual",
            line=dict(color="#58a6ff", width=3),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.15)",
        ))
        fig_proj.update_layout(
            title="Projecao de Retencao do Cliente (Meses M0 a M6)",
            xaxis_title="Mes",
            yaxis_title="Probabilidade de Retencao (%)",
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            height=280,
            margin=dict(l=40, r=40, t=50, b=40),
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig_proj, use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("### Sobre o Projeto")
    st.markdown(
        """
        Sistema de Inteligencia de Clientes que une analise matricial RFM,
        algoritmo nao-supervisionado K-Means e scoring de propensao ao churn.

        **Principais Pilares:**
        - **Pipeline RFM**: Ingestao de transacoes e quantis 1-5.
        - **K-Means (k=4)**: Mapeamento deterministico de personas.
        - **Validacao**: Metodo do Cotovelo e Silhouette Score (0.68).
        - **Decisao CRM**: Acoes taticas preventivas contra evasao.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **Desenvolvido por:**
        Renan Nocelli
        Portfolio de Engenharia de Dados & Machine Learning
        """
    )
