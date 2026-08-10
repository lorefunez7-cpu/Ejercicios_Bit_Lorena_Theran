import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Proyecto Final ML - Women's Clothing Reviews", layout="wide")


@st.cache_data
def cargar_datos():
    df = pd.read_csv("dashboard_datos.csv")
    comparacion = pd.read_csv("dashboard_comparacion_modelos.csv")
    df_nlp = pd.read_csv("dashboard_texto_nlp.csv")
    return df, comparacion, df_nlp


df, comparacion, df_nlp = cargar_datos()

st.title("Proyecto Final – Machine Learning en Reseñas de E-Commerce")
st.markdown("Análisis de reseñas de clientas de una tienda de ropa online, combinando Aprendizaje Supervisado, No Supervisado y NLP.")

# ---------- Filtros ----------
st.sidebar.header("Filtros")
departamentos_disponibles = sorted(df["Department Name"].dropna().unique().tolist())
departamentos = st.sidebar.multiselect(
    "Departamento", options=departamentos_disponibles, default=departamentos_disponibles
)
df_filtrado = df[df["Department Name"].isin(departamentos)] if departamentos else df

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de reseñas", f"{len(df_filtrado):,}")
col2.metric("Calificación promedio", f"{df_filtrado['Rating'].mean():.2f} / 5")
col3.metric("% Recomienda", f"{df_filtrado['Recommended IND'].mean() * 100:.1f}%")
col4.metric("Edad promedio", f"{df_filtrado['Age'].mean():.0f} años")

st.divider()

tab1, tab2, tab3 = st.tabs(["Aprendizaje Supervisado", "Aprendizaje No Supervisado", "NLP"])

# ---------- Ruta A: Supervisado ----------
with tab1:
    st.subheader("Comparación de modelos")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(comparacion.set_index("Modelo"))
    with col2:
        comparacion_larga = comparacion.melt(id_vars="Modelo", var_name="Metrica", value_name="Valor")
        fig, ax = plt.subplots()
        sns.barplot(data=comparacion_larga, x="Metrica", y="Valor", hue="Modelo", palette="Set2", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title("Comparación de métricas por modelo")
        st.pyplot(fig)

    mejor_modelo = comparacion.set_index("Modelo")["F1-score"].idxmax()
    st.success(f"Mejor modelo según F1-score: **{mejor_modelo}**")

    st.subheader("Distribución de calificaciones")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtrado, x="Rating", hue="Rating", palette="viridis", legend=False, ax=ax2)
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Cantidad de reseñas")
    st.pyplot(fig2)

# ---------- Ruta B: No Supervisado ----------
with tab2:
    st.subheader("Segmentación de clientas (K-Means)")

    if "cluster_kmeans" in df_filtrado.columns:
        perfil_clusters = df_filtrado.groupby("cluster_kmeans")[["Age", "Rating", "Positive Feedback Count"]].mean()
        tamano_clusters = df_filtrado["cluster_kmeans"].value_counts().sort_index()

        col1, col2 = st.columns(2)
        with col1:
            st.write("Tamaño de cada grupo")
            fig3, ax3 = plt.subplots()
            sns.barplot(
                x=tamano_clusters.index.astype(str),
                y=tamano_clusters.values,
                hue=tamano_clusters.index.astype(str),
                palette="mako",
                legend=False,
                ax=ax3,
            )
            ax3.set_xlabel("Cluster")
            ax3.set_ylabel("Cantidad de reseñas")
            st.pyplot(fig3)
        with col2:
            st.write("Perfil promedio por grupo")
            st.dataframe(perfil_clusters)
    else:
        st.warning("No se encontró la columna 'cluster_kmeans' en los datos exportados.")

# ---------- Ruta C: NLP ----------
with tab3:
    st.subheader("Palabras más frecuentes en las reseñas")

    texto_pos = " ".join(df_nlp[df_nlp["Recommended IND"] == 1]["texto_limpio"].dropna().astype(str))
    texto_neg = " ".join(df_nlp[df_nlp["Recommended IND"] == 0]["texto_limpio"].dropna().astype(str))

    col1, col2 = st.columns(2)
    with col1:
        st.write("Reseñas que SÍ recomiendan")
        if texto_pos.strip():
            wc1 = WordCloud(width=500, height=350, background_color="white", colormap="Greens").generate(texto_pos)
            fig4, ax4 = plt.subplots()
            ax4.imshow(wc1, interpolation="bilinear")
            ax4.axis("off")
            st.pyplot(fig4)
    with col2:
        st.write("Reseñas que NO recomiendan")
        if texto_neg.strip():
            wc2 = WordCloud(width=500, height=350, background_color="white", colormap="Reds").generate(texto_neg)
            fig5, ax5 = plt.subplots()
            ax5.imshow(wc2, interpolation="bilinear")
            ax5.axis("off")
            st.pyplot(fig5)

st.divider()
st.caption("Proyecto Final - Aplicación de Machine Learning a un Caso Real | Dataset: Women's E-Commerce Clothing Reviews")
