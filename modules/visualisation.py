import pandas as pd
import plotly.express as px

def make_plot(df: pd.DataFrame, cols: list[str]):
    if not cols:
        return None
    # 1 colonne
    if len(cols) == 1:
        col = cols[0]
        return (
            px.histogram(df, x=col, nbins=30)
            if pd.api.types.is_numeric_dtype(df[col])
            else px.bar(df[col].value_counts().reset_index(), x="index", y=col)
        )
    # 2 colonnes
    if len(cols) == 2:
        c1, c2 = cols
        if all(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
            return px.scatter(df, x=c1, y=c2)
        return px.bar(df, x=c1, y=c2)
    # >2 colonnes
    return px.scatter_matrix(df[cols])

def line_by_country(df: pd.DataFrame,
                    year_col: str,
                    value_col: str,
                    country_col: str):
    """
    Trace l'évolution de value_col en fonction de year_col,
    colorié par country_col, SANS convertir year_col en datetime.
    """
    # === 1. Sous-dataframe propre ===
    d = df[[year_col, value_col, country_col]].copy()
    # Conversion sûre en numérique (ignore erreurs -> NaN) puis drop NaN
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d = d.dropna(subset=[year_col, value_col, country_col])

    # === 2. Tri chronologique ===
    d = d.sort_values(year_col)

    # === 3. Tracé ===
    fig = px.line(
        d,
        x=year_col,           # Numérique : axe continu
        y=value_col,
        color=country_col,
        markers=True
    )
    fig.update_layout(
        xaxis_title=year_col,
        yaxis_title=value_col,
        legend_title=country_col,
        xaxis=dict(type="linear", tickformat="d")  # 'd' = entier
    )
    return fig