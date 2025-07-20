import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from tslearn.metrics import soft_dtw

from modules.data_loader import load_dataset
from modules.analyse import description_stats
from modules.visualisation import make_plot, line_by_country

from modules.profilage.profilage_1 import dataset_profile1
from modules.profilage.profilage_2 import analyse_profilage2,calcul_centroids,centroid_unique
from modules.calculateur.calculateur import two_step_interpolation, sd_distance
import tempfile

import matplotlib.pyplot as plt
import io
from PIL import Image

st.set_page_config(page_title="Plateforme d’analyse", layout="wide")

# ───────────────────────────────
# Chargement conditionnel avec ou sans défaut
# ───────────────────────────────
def charger_ou_memo(nom_etat, fichier, use_default=False, mode="donnees"):
    if fichier:
        sheets, path = load_dataset(fichier)
        st.session_state[nom_etat] = {"sheets": sheets, "path": path, "file"  : fichier }
        return sheets, path

    if nom_etat in st.session_state:
        return st.session_state[nom_etat]["sheets"], st.session_state[nom_etat]["path"]

    if use_default:
        sheets, path = load_dataset(None, mode=mode)
        st.session_state[nom_etat] = {"sheets": sheets, "path": path, "file"  : None }
        return sheets, path

    return None, None

# ───────────────────────────────
# MENU principal
# ───────────────────────────────
menu = st.radio("Navigation", ["Données", "Profilage des pays", "Calculateur de trajectoire"], horizontal=True)

# ───────────────────────────────
# SIDEBAR – upload spécifique
# ───────────────────────────────
# ────────────────────────────── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.title("📂 Fichiers à utiliser")

    # ---------- Onglet Données ----------
    if menu == "Données":
        file = st.file_uploader("Fichier pour Données", type=["xlsx"], key="file_donnees")
        sheets, path = charger_ou_memo("df_donnees", file, use_default=True, mode="donnees")

    # ---------- Onglet Profilage des pays ----------
    elif menu == "Profilage des pays":
        st.markdown("### 🔎 Profilage 1 (CAH-DTW)")
        file_p1 = st.file_uploader("Upload Profilage 1", type=["xlsx"], key="file_profilage1")
        sheets_p1, path_p1 = charger_ou_memo("df_profilage1", file_p1, use_default=True, mode="profilage1")

        st.markdown("---")
        st.markdown("### 🧩 Profilage 2 (Soft-DTW multivarié)")
        file_p2 = st.file_uploader("Upload Profilage 2", type=["xlsx"], key="file_profilage2")
        sheets_p2, path_p2 = charger_ou_memo("df_profilage2", file_p2, use_default=True, mode="profilage2")

        # Petit résumé
        if sheets_p1:
            st.caption(f"✅ Profilage 1 chargé : {path_p1}")
        else:
            st.info("Profilage 1 : aucun fichier sélectionné.")
        if sheets_p2:
            st.caption(f"✅ Profilage 2 chargé : {path_p2}")
        else:
            st.info("Profilage 2 : aucun fichier sélectionné.")

    # ---------- Onglet Calculateur ----------
    else:
        file = st.file_uploader("Fichier pour Calculateur", type=["xlsx"], key="file_calc")
        sheets, path = charger_ou_memo("df_calculateur", file, use_default=True, mode="calculateur")

        if sheets:
            st.caption(f"✅ Fichier chargé : {path}")
        else:
            st.info("Aucun fichier encore sélectionné.")
        
        with st.expander("📥 Télécharger le fichier Excel par défaut"):
            with open("data/sample_pays_avant_covid.xlsx", "rb") as f:
                st.download_button(
                    label="📥 Télécharger sample_pays_avant_covid_.xlsx",
                    data=f,
                    file_name="sample_pays_avant_covid.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ───────────────────────────────
# PAGE : DONNÉES
# ───────────────────────────────
if menu == "Données":
    st.header("📁 Données")

    if sheets:
        df = sheets[st.selectbox("Feuille / tab Excel", list(sheets.keys()), key="tab_donnees")]

        with st.expander("🔎 Aperçu"):
            st.dataframe(df.head())

        with st.expander("📈 Statistiques descriptives"):
            st.dataframe(description_stats(df))

        st.subheader("📊 Visualisation")

        vis_type = st.radio("Type de graphique", ["Générique", "Évolution temporelle"], horizontal=True)

        fig = None
        try:
            if vis_type == "Générique":
                cols = st.multiselect("Colonnes à tracer", df.columns)
                if cols:
                    fig = make_plot(df, cols)
            else:
                year = st.selectbox("Colonne Années", df.columns)
                value = st.selectbox("Indicateur", df.select_dtypes('number').columns)
                country = st.selectbox("Pays", df.select_dtypes(exclude='number').columns)
                if year and value and country:
                    fig = line_by_country(df, year, value, country)
        except Exception as e:
            st.error(f"Erreur : {e}")

        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez déposer un fichier ou utiliser celui par défaut.")

# ───────────────────────────────
# PAGE : PROFILAGE
# ───────────────────────────────
# ───────────────────────────────
# PAGE : PROFILAGE DES PAYS
# ───────────────────────────────

elif menu == "Profilage des pays":
    profilage1_state = st.session_state.get("df_profilage1", None)
    profilage2_state = st.session_state.get("df_profilage2", None)

    if profilage1_state:
        st.header("🌍 Profilage des pays")
        tab1, tab2 = st.tabs(["📌 Profilage 1", "📊 Profilage 2"])

        # ------------- PROFILAGE 1 ------------------------------------------
        with tab1:
            st.subheader("Profilage 1 : Évolution des profils de pays avec le temps")

            file_path  = profilage1_state["path"]
            uploaded_file_p1 = profilage1_state.get("file")  # objet UploadedFile
            sheets     = profilage1_state["sheets"]
            sheet_name = st.selectbox("Feuille Excel", list(sheets.keys()), key="tab_profilage1")
            df1 = sheets[sheet_name]
            st.dataframe(df1.head())

            nb_years = st.number_input("Durée de chaque segment (années)", min_value=2, max_value=10, value=5, step=1)

            # Choix de la source : objet mémoire si dispo, sinon chemin par défaut
            file_source_p1 = uploaded_file_p1 if uploaded_file_p1 else file_path

            if st.button("Lancer l’analyse CAH"):
                try:
                    clusters_df = dataset_profile1(file_source_p1, sheet_name=sheet_name, nb_years=nb_years)
                    clusters_df.columns = clusters_df.columns.map(str)
                    st.session_state["clusters_df"] = clusters_df
                    st.session_state["profilage1_sheet"] = sheet_name
                    st.success("Clustering effectué avec succès.")
                    st.subheader("Tableau des groupes par segment")
                    st.dataframe(clusters_df)

                except Exception as e:
                    st.error(f"Erreur durant l’analyse : {e}")

            clusters_df = st.session_state.get("clusters_df")
            if clusters_df is not None:
                st.subheader("Evolution des pays par période")
                fig = px.imshow(
                    clusters_df,
                    color_continuous_scale="viridis",
                    labels=dict(x="Segment", y="Pays", color="Cluster"),
                    aspect="auto",
                )
                st.plotly_chart(fig, use_container_width=True)

                #st.subheader("Répartition des clusters par segment")
                #counts = clusters_df.apply(lambda col: col.value_counts()).T
                #fig = px.bar(counts, barmode="relative")
                #st.plotly_chart(fig, use_container_width=True)

                csv = clusters_df.to_csv().encode("utf-8")
                st.download_button("🗵️ Télécharger CSV", csv, "clusters.csv", "text/csv")
            else:
                st.info("Lance l’analyse pour générer les clusters.")

        # ------------- PROFILAGE 2 ------------------------------------------
        with tab2:
            st.subheader("Profilage 2 : Impact et réaction face aux crises")

            if not profilage2_state or profilage2_state["sheets"] is None:
                st.info("Charge d’abord un fichier Profilage 2 dans la barre latérale.")
            else:
                uploaded_file = profilage2_state["file"]
                nb_groups = st.number_input("🔹 Nombre de groupes", 2, 10, 3, 1, key="profilage2_ng")

                # --- bouton : recalcul ---
                if st.button("🚀 Lancer l’analyse Profilage 2", key="profilage2_run"):
                    uploaded_file = profilage2_state["file"]
                    file_path     = profilage2_state["path"]
                    file_or_upload = uploaded_file if uploaded_file else file_path

                    resultats = analyse_profilage2(file_or_upload, n_groups=nb_groups)
                    #resultats = analyse_profilage2(uploaded_file, n_groups=nb_groups)

                    if not resultats:
                        st.warning("Aucun résultat généré. Vérifie le contenu du fichier.")
                    else:
                        st.session_state["profilage2_resultats"] = {
                            "k": nb_groups,
                            "data": resultats
                        }
                        st.session_state["centroids_profilage2"] = {}

                res_state = st.session_state.get("profilage2_resultats", None)
                resultats = res_state["data"] if res_state and res_state.get("k") == nb_groups else None

                if resultats:
                    noms_feuilles = list(resultats.keys())
                    onglets = st.tabs(noms_feuilles)

                    for onglet, feuille in zip(onglets, noms_feuilles):
                        fig, df_clusters = resultats[feuille]

                        with onglet:
                            st.markdown(f"### 📄 Feuille : **{feuille}**")
                            st.pyplot(fig)

                            grouped = df_clusters.groupby("Cluster")["Pays"].apply(list)
                            max_len = grouped.str.len().max()
                            padded = grouped.apply(lambda x: x + [""] * (max_len - len(x)))
                            tbl_cls = pd.DataFrame(padded.tolist()).T
                            tbl_cls.columns = [f"Cluster {c}" for c in grouped.index]
                            tbl_cls.index = range(1, max_len + 1)

                            st.subheader("📊 Répartition des pays par cluster")
                            st.dataframe(tbl_cls, use_container_width=True)

                            if "centroids_profilage2" not in st.session_state:
                                st.session_state["centroids_profilage2"] = {}
                            centroids_cache = st.session_state["centroids_profilage2"]

                            if feuille in centroids_cache:
                                centroids, years, factors = centroids_cache[feuille]
                            else:
                                uploaded_file = profilage2_state["file"]
                                file_path     = profilage2_state["path"]
                                file_or_upload = uploaded_file if uploaded_file else file_path

                                # Cas fichier uploadé : sauvegarder temporairement pour lecture
                                if hasattr(file_or_upload, "read"):
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                                        tmp.write(file_or_upload.read())
                                        tmp_path = tmp.name
                                else:
                                    tmp_path = str(file_or_upload)

                                df_sheet = pd.read_excel(tmp_path, sheet_name=feuille, engine="openpyxl")

                                #df_sheet = pd.read_excel(uploaded_file, sheet_name=feuille, engine="openpyxl")
                                centroids, years, factors = calcul_centroids(df_sheet, df_clusters)
                                # —— Normalisation : supprime la 1ère dim inutile
                                for cid, mat in centroids.items():
                                    if mat.ndim == 3 and mat.shape[0] == 1:
                                        centroids[cid] = np.squeeze(mat, axis=0)   # -> (n, v)
                                centroids_cache[feuille] = (centroids, years, factors)

                            with st.expander("📍 Pays répresentatifs de clusters"):
                                sub_tabs = st.tabs([f"Cluster {cid}" for cid in sorted(centroids)])

                                for sub_tab, cid in zip(sub_tabs, sorted(centroids)):
                                    with sub_tab:
                                        c_data = centroids[cid]

                                        if c_data.ndim == 3 and c_data.shape[0] == 1:
                                            c_data = np.squeeze(c_data, axis=0)
                                        elif c_data.ndim == 1:
                                            c_data = c_data.reshape(-1, 1)

                                        n_years, n_fac = c_data.shape
                                        cols = [factors[i] for i in range(n_fac)]
                                        c_mat = pd.DataFrame(c_data, index=years[:n_years], columns=cols)

                                        st.write(f"pays réprésentatif – Cluster {cid}")
                                        st.dataframe(c_mat.round(3), use_container_width=True)

                            csv = df_clusters.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                f"🗕️ Télécharger CSV ({feuille})",
                                csv,
                                file_name=f"clusters_{feuille}.csv",
                                mime="text/csv",
                                key=f"dl_{feuille}",
                            )
                
                # ───────────── Affichage style "comparaison des profils" ─────────────
                st.markdown("## 📊 Comparaison des niveaux (Avant – Pendant – Après) par pays")

                try:
                    all_results = st.session_state.get("profilage2_resultats", {}).get("data", {})
                    
                    # Map réel → étiquette affichée
                    mapping = {
                        "avant_crise": "Avant",
                        "Pendant_covid": "Pendant",
                        "Apres_covid": "Après"
                    }

                    feuilles_disponibles = list(all_results.keys())

                    # Vérifie que toutes les clés nécessaires sont bien présentes
                    if all(k in feuilles_disponibles for k in mapping.keys()):
                        df_concat = pd.DataFrame()

                        for feuille, periode in mapping.items():
                            df_clusters = all_results[feuille][1]
                            df_tmp = df_clusters[["Pays", "Cluster"]].copy()
                            df_tmp["Période"] = periode
                            df_concat = pd.concat([df_concat, df_tmp], ignore_index=True)

                        # Pivot: index = Période, colonnes = Pays, valeurs = Cluster
                        df_plot = df_concat.pivot(index="Période", columns="Pays", values="Cluster").T
                        df_plot = df_plot[["Avant", "Pendant", "Après"]]  # assure l’ordre

                        #import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(df_plot.index, df_plot["Avant"], color="deepskyblue", marker='o', label="Avant")
                        ax.plot(df_plot.index, df_plot["Pendant"], color="limegreen", marker='o', label="Pendant")
                        ax.plot(df_plot.index, df_plot["Après"], color="gray", marker='o', label="Après")

                        ax.set_ylim(0, 4)
                        ax.set_yticks([1, 2, 3])
                        ax.set_ylabel("Niveau de cluster")
                        ax.set_xticks(range(len(df_plot.index)))
                        ax.set_xticklabels(df_plot.index, rotation=60)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        ax.legend()
                        ax.set_title("Évolution des profils des pays durant les trois périodes – MIRACE")

                        st.pyplot(fig)

                        # Sauvegarde temporaire en mémoire
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        st.download_button("📥 Télécharger le graphique PNG", buf, file_name="comparaison_niveaux.png", mime="image/png")

                    else:
                        st.warning(f"Feuilles manquantes. Requises : {list(mapping.keys())} ; présentes : {feuilles_disponibles}")

                except Exception as e:
                    st.error(f"Erreur lors du tracé comparatif : {e}")


    else:
        st.info("Veuillez charger un fichier pour le profilage.")

# ───────────────────────────────
# PAGE : CALCULATEUR
# ───────────────────────────────
# ===================== CALCULATEUR DE TRAJECTOIRE =====================
elif menu == "Calculateur de trajectoire":
    st.header("📈 Calculateur de trajectoire – Attribution de profil")

    # ─── 1) Vérifie que les centroïdes du Profilage 2 existent ───
    centroids_state = st.session_state.get("centroids_profilage2", None)
    if not centroids_state:
        st.info("Veuillez d’abord exécuter Profilage 2 pour générer les centroïdes.")
        st.stop()

    # ─── 2) Vérifie qu’un fichier CSV a été chargé dans la sidebar ───
    calc_state = st.session_state.get("df_calculateur", None)
    if not calc_state or not calc_state["sheets"]:
        st.info("Chargez un EXCEL dans la barre latérale pour le calculateur.")
        st.stop()

    # On récupère la première (et unique) “feuille” puisque c’est un CSV
    df_pays = list(calc_state["sheets"].values())[0]
    st.success("Données pays chargées :")
    st.dataframe(df_pays.head())

    # ─── 3) Choix de la période (feuille) disponible dans centroids_state ───
    periods = list(centroids_state.keys())
    period = st.selectbox("Période de référence", periods)

    # ─── 4) Extraction de la série temporelle du pays ───
    # (on suppose que la première colonne du CSV est l'année, le reste = variables)
    new_series = df_pays.iloc[:, 1:].values          # shape (n_années, n_vars)
    # --- DEBUG : shapes ----
    #if st.checkbox("Afficher les shapes (debug)", key="chk_shapes"):
    #    st.write("Shape new_series :", new_series.shape)

    # ─── 5) Calcul Soft-DTW vers chaque centroïde ───
    gamma = 1.0
    distances = {}
    centroids_dict, years, factors = centroids_state[period]  # tuple stocké

    nb_vars_pays = new_series.shape[1]

    for cid, centroid in centroids_dict.items():
        #if st.session_state.get("chk_shapes"):
        #    st.write(f"Centre {cid} shape :", centroid.shape)
        nb_vars_centroid = centroid.shape[1]
        nb_common = min(nb_vars_pays, nb_vars_centroid)

        d_tot = 0.0
        for j in range(nb_common):
            x = new_series[:, j]   # vecteur 1-D
            y = centroid[:, j]
            d_tot += soft_dtw(x, y, gamma=gamma)

        # pénalise si le nombre de variables diffère
        if nb_vars_pays != nb_vars_centroid:
            st.write("nombre de variables different")
            d_tot += 1e6

        distances[cid] = d_tot

    # ─── 6) Attribution du cluster le plus proche ───
    best_cluster = min(distances, key=distances.get)

    st.subheader(f"🔎 Attribution pour la période {period}")
    st.write(distances)
    st.success(f"👉 Le pays est attribué au **Cluster {best_cluster}**")
    
    
    st.subheader(f"🔎 Simulateur : Comportement du pays à l’arrivée de la crise")
    # --------------------------------------------------------------------
    # 2) Paramètre k et sélection des deux feuilles
    # --------------------------------------------------------------------
    #st.markdown("---")
    #st.subheader("🎯 Construction d’un jeu concaténé (Avant ↦ Pendant)")

    k_years = st.number_input(
        "k : nombre d’années à récupérer depuis la feuille Avant",
        min_value=1, max_value=30, value=5, step=1
    )

    profil2_state = st.session_state.get("profilage2_resultats", {})
    if not profil2_state:
        st.warning("Aucun résultat Profilage 2 en mémoire – exécute d’abord Profilage 2.")
    else:
        sheets_all = list(profil2_state["data"].keys())
        sel_sheet = st.selectbox("Période de projection", periods)
        #sel_sheets = st.multiselect(
        #    "Choisis exactement 2 feuilles (ex.: Avant, Pendant)",
        #    options=sheets_all,
        #    default=sheets_all[:2],          # premières par défaut
        #    max_selections=2
        #)

        #sheet_av, sheet_pd = sel_sheets
        sheet_av = period
        sheet_pd = sel_sheet

        if st.button("Simuler"):
            try:
                #file_p2   = st.session_state["df_profilage2"]["file"]
                
                profilage2_state = st.session_state.get("df_profilage2", {})
                uploaded_file_p2 = profilage2_state.get("file")
                path_p2          = profilage2_state.get("path")

                file_p2 = uploaded_file_p2 if uploaded_file_p2 else path_p2
                if hasattr(file_p2, "read"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                        tmp.write(file_p2.read())
                        tmp_path = tmp.name
                else:
                    tmp_path = str(file_p2)


                # DataFrame clusters de la feuille Avant
                df_cl_av  = profil2_state["data"][sheet_av][1]   # index 1 = df_clusters
                pays_sel  = df_cl_av[df_cl_av["Cluster"] == best_cluster]["Pays"].tolist()

                df_avant   = pd.read_excel(tmp_path, sheet_name=sheet_av, engine="openpyxl")
                df_pendant = pd.read_excel(tmp_path, sheet_name=sheet_pd, engine="openpyxl")

                years_kept      = sorted(df_avant["Années"].unique())[-int(k_years):]
                first_year_pd   = sorted(df_pendant["Années"].unique())[0]

                subset_avant = df_avant[
                    (df_avant["Années"].isin(years_kept)) & (df_avant["Pays"].isin(pays_sel))
                ]
                subset_pend  = df_pendant[
                    (df_pendant["Années"] == first_year_pd) & (df_pendant["Pays"].isin(pays_sel))
                ]

                df_concat = pd.concat([subset_avant, subset_pend], ignore_index=True)
                
                # ── Fusion des colonnes R_AV_COV & R_P_COV -> R_COV ──
                if "R_P_COV" in df_concat.columns and "R_AV_COV" in df_concat.columns:
                    df_concat["R_COV"] = df_concat["R_P_COV"].combine_first(df_concat["R_AV_COV"])
                    df_concat.drop(columns=["R_P_COV", "R_AV_COV"], inplace=True)
                    df_concat = df_concat.rename(columns={'R_COV': 'R_P_COV'})

                #st.subheader("📅 Jeu de données concaténé")
                #st.dataframe(df_concat, use_container_width=True)

                #csv_concat = df_concat.to_csv(index=False).encode("utf-8")
                #st.download_button("💾 Télécharger CSV concaténé",csv_concat,file_name="concat.csv",mime="text/csv")
                
                #st.subheader(f"🔎 Calcul du comportement en {first_year_pd} - Debut crise")
                # après avoir construit df_concat
                centroid, years, factors = centroid_unique(df_concat)
                
                val_2020 = (
                    pd.DataFrame(centroid, index=years, columns=factors)
                        .loc[2020]     # sélectionne la ligne 2020
                        .round(3)        # arrondit
                )
                st.session_state["est_val_2020"] = val_2020
                
                st.dataframe(val_2020, use_container_width=True)
                
                

            except Exception as e:
                st.error(f"Erreur lors de la construction : {e}")
        
        cluster_target = st.selectbox("Le profil cible", centroids_dict)
        # ── 8) Préparation projection 2021 selon deux-étapes Soft-DTW ───────────
        #eps_slider = st.slider("Tolérance Soft-DTW ε", 0.05, 1.0, 0.20, 0.05)
        eps_slider = 0.05
        
        if st.button("Estimer"):
            centroids_dict, years, factors = centroids_state[sheet_pd]  # tuple stocké
            
            # 3. Extraire centroïde du cluster choisi
            centroid_df = pd.DataFrame(centroids_dict[cluster_target], index=years, columns=factors)
            #st.dataframe(centroid_df)
            
            # 3️⃣ Calcul des bornes min / max pour chaque variable du profil
            df_clusters  = profil2_state["data"][sheet_pd][1]   # index 1 = df_clusters
            pays_du_cluster = df_clusters[df_clusters["Cluster"] == cluster_target]["Pays"].tolist()
            
            # 2) création du DataFrame de bornes
            # 3️⃣ Calcul des bornes min / max dans le cluster cible entre 2020 et 2021

            bounds = pd.DataFrame({
                "min": centroid_df[factors].min(),   # ligne 'min'
                "max": centroid_df[factors].max()    # ligne 'max'
            }).T  
            
            # 5. Interpolation  est_val_2020
            val_2020 = st.session_state["est_val_2020"]
            #st.dataframe(val_2020)
            
            v0_adj, v1_pred, alpha, beta = two_step_interpolation(
                val_2020  = val_2020[factors],
                c_2020    = centroid_df.loc[2020],
                c_2021    = centroid_df.loc[2021],
                bounds    = bounds,
                eps       = eps_slider
            )
            # 6. Affichage
            st.dataframe(
                pd.DataFrame([v0_adj, v1_pred], index=[2020, 2021]).T,
                use_container_width=True
            )
            st.caption(f"α = {alpha:.3f}   β = {beta:.3f}  (Soft-DTW ≤ {eps_slider})")
                        
            #centroid_df = pd.DataFrame(centroid, index=years, columns=factors)
            
            #factors = df_concat.columns.difference(["Pays", "Année"])
                


else:
    st.header("📈 Calculateur de trajectoire")

    if sheets:
        df = sheets[st.selectbox("Feuille", list(sheets.keys()), key="tab_calc")]
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
    else:
        st.info("Veuillez charger un fichier pour commencer.")
