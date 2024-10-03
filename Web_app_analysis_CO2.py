# Import des bibliothèques
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np
import pickle
import os
from PIL import Image
import base64

# Import des données nécessaires
@st.cache_data
def import_dataframe():
    df = pd.read_csv('Data/data_head.csv', sep=',', low_memory=False)
    df_non_pertinent = pd.read_csv('Data/df_head_nonpertinent.csv', sep=',', low_memory=False)
    df_final = pd.read_csv('Data/df_head_final.csv', sep=',', low_memory=False)
    return df,df_non_pertinent, df_final
df, df_non_pertinent, df_final = import_dataframe()

# Création des pages et de la sidebar
pages = ['Introduction', 'Exploration des données', 'DataViz\'', 'Machine Learning', 'Dashboard', 'Conclusion et perspectives']
st.sidebar.write('# Emission CO2')
page = st.sidebar.radio('Aller vers', pages)
st.sidebar.write('# Auteurs')
st.sidebar.write('Romain NALLET')
st.sidebar.write('Valentin CHARRIER')
st.sidebar.write('Soulaimana YSSOUFI')
st.title('Emission CO2')

# Page 1
if page == 'Introduction':
    st.image("https://ispira-qualite-air.fr/wp-content/uploads/2021/08/pollution-de-lair-en-ville.jpeg")
    st.header('Introduction')
    st.subheader('Contexte')
    st.write("Dans le cadre de la formation 'Data Analyst' que nous avons tous suivis au sein de Datascientest, nous avons travaillé sur un projet s\'étendant sur l\'ensemble de la période de formation.\n\n Dans le contexte de notre formation, nous avons pu appliquer nos connaissances au fur et à mesure de notre progression sur un cas concret d\'analyse de données,\
              et ainsi améliorer nos compétences.")

    st.subheader('Problématique')
    st.write("Lors de la constitution de notre groupe, nous avons tous souhaité que le projet soit orienté vers le développement d'un modèle de machine Learning afin de pouvoir prédire les émissions de CO² \
             d'un véhicule en fonction de différents paramètres (par exemple : type de motorisation, poids, consommation, etc...).")

    st.subheader('Sources de données')
    st.write("Lors de l\'attribution desprojets, nous avons eu à notre disposition deux sources de données différentes. Une provenant de l\'[EEA](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b) (European Environment Agency)\
             et l\'autre du [gouvernement français](https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/#_).\n\nNous avons choisi d\'utiliser les données provenant de l\'EEA pour l\'année 2023 car cette source contient beaucoup plus de données.")

# Page 2
if page == 'Exploration des données':
    st.header('Exploration des données')

    st.subheader('Visualisation du DataFrame')
    st.dataframe(df.head(5))
    st.write("Le DataFrame comprend 5 751 790 lignes et 39 colonnes")

    st.subheader('Distribution de la variable cible')
    col1, col2 = st.columns([8,1])
    with col1 :
        button2_result = st.button("Retirer les véhicules électriques/hybrides")
        if button2_result :   
            st.image('Images/image_Ewltp_streamlit_2.png', caption = 'Pour les véhicules thermiques seuls')
            st.write('Ce graphique montre une forte répartition des données entre 100 et 200 g/km d’émission de CO² de la variable \'Ewltp (g/km)\',\
                 pour les véhicules thermiques')
        else :
            st.image('Images/image_Ewltp_streamlit_1.png', caption = 'Pour les véhicules thermiques et hybrides')
            st.write('Ce graphique montre 2 pics, qui représentent les véhicules électriques/hybrides (moins polluant) et les véhicules thermiques (plus polluants).')
    with col2 :    
        button3_result = st.button("Reset")

    st.subheader('Distribution de Fuel_type/Fuel_mode')
    st.image("Images/image_Fuel_mode_et_Fuel_type_streamlit.png")
    st.write("A l\'aide de ce graphique nous pouvons observer que nos 2 variables 'Fuel_mode' et 'Fuel_cons' sont interprétables de façon commune.\
        Ainsi, il est possible de générer des associations comme suit :")
    st.write("- Les véhicules électriques/hyrdogènes qui sont associées au mode de carburant E")
    st.write("- Les véhicules hybrides qui sont associées au mode de carburant P ")
    st.write("- Les véhicules thermiques qui sont associées aux modes de carburant M, B, H et F")
    st.write("Pour plus d'informations consulter [cette page](https://circabc.europa.eu/sd/a/d9cff59f-5117-48f4-9a37-07b94027110c/MS%20Guidelines%202019).") 
    st.write(" Nous avons donc 3 grands types de motorisations.")
    st.markdown("*Note : Sachant que les hybrides non rechargeables (mode de carburant H) sont considérés comme des véhicules thermiques.*")

    st.subheader('Gestion des colonnes à valeurs nulles')
    st.write("Lors de l'exploration de notre Dataset, nous avons généré une matrice de chaleur présentant les valeurs nulles dans chaque colonnes du DataFrame")
    col1, col2 = st.columns([8,1])
    with col1 :
        button4_result = st.button("Retirer les colonnes avec uniquement des valeurs nulles")
        if button4_result :   
            st.image('Images/image_matrice_chaleur_nulle_streamlit_2.png', caption = "Matrice de chaleur")
        else :
            st.image('Images/image_matrice_chaleur_nulle_streamlit_1.png', caption = "Matrice de chaleur")
    with col2 :    
        button5_result = st.button("Reset ")
    st.write("A l'aide de cette matrice nous avons pu identifier faciement les colonnes présentant uniquement des valeurs nulles.\
        Nous avons décidé de supprimer les colonnes concernés.")
    st.code('''# Suppression des colonnes aux valeurs manquantes :
df.drop(['MMS', 'Enedc (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ernedc (g/km)',
'De', 'Vf'], axis = 1, inplace = True)''')
    st.write("Le DataFrame se compose alors de 5 751 790 lignes et 31 colonnes.")
    
    st.subheader('Gestion des colonnes à valeurs non pertinentes')
    st.write("La poursuite de l'exploration du Dataset, nous a amené à constater la présence de colonnes qui ne contienent que des valeurs non pertinentes pour notre analyse")
    st.write("*Par exemple : la colonne 'VFN', contenant des suites de chiffres et de lettres sans logique apparente.*")
    st.code('''# Suppression des colonnes aux valeurs non pertinentes pour l'analyse :
df.drop(['VFN','IT', 'Date of registration', 'RLFI', 'Tan', 'Erwltp (g/km)',
'ech'], axis = 1, inplace = True)''')
    st.dataframe(df_non_pertinent.head(5))
    st.write("Le DataFrame se compose alors de 5 751 790 lignes et 24 colonnes.")

    st.subheader('Gestion des colonnes à valeurs unique')
    st.write("En continuant notre exploration, nous avons également remarqué la présence de colonnes avec des données uniques.\
             *Par exemple, la colonne 'Year' qui ne contient que la valeur '2023'.*\
             Nous, avons choisi de supprimer ces colonnes également car elle ne présentaient pas d'interêt pour notre analyse")
    st.code('''# Suppression des colonnes à valeurs uniques :
df.drop(['r','Status', 'year'], axis = 1, inplace = True)''')
    st.dataframe(df_final.head(5))
    st.write("Le DataFrame se compose alors de 5 751 790 lignes et 21 colonnes.")
    
    st.subheader("DataFrame en fin d'exploration")
    st.write("Finalement, Nous décidons de supprimer les véhicules électriques/hydrogènes de notre DataFrame, car ceux-ci n'émettent pas de CO².\
            Le DataFrame comprends donc au final : 4 983 281 lignes et 21 colonnes.")
    
# Page 3
if page == 'DataViz\'':
    st.header('Dataviz\'')
    st.subheader('Variables numériques')
    st.write('Dans cette partie nous nous attachons à observer les liens entre nos variables numériques.')
    st.write('#### Matrice de corrélation')
    Aff_therm = st.checkbox('Véhicules thermique uniquement')
    if Aff_therm == False:
        st.image('Images/Heatmap.png')
    else:
        st.image('Images/Heatmap_thermique.png')
    st.write('Les niveaux de corrélation avec la variable cible sur l\'ensemble des données sont assez faibles.')
    st.write('Par contre, les niveaux sont bien plus élevés lorsque l\'on ne regarde que la catégorie des véhicules thermiques :')
    st.write('* Fuel_cons : 0.93\
             \n* Mass : 0.74\
             \n* Horse_power et Engin_capacity : 0.77 et 0.73')
    st.write('#### Relation entre emission CO2 et la consommation de carburant')
    st.write('Comme on pourrait s\'y attendre, un lien très fort existe entre la consommation de carburant et le rejet CO2. Regardons cette relation de plus près.')
    Zoom_plateau = st.checkbox('Zoom sur le plateau')
    if Zoom_plateau == False:
        st.image('Images/Relation_fuel_cons_Ewltp_global.png')
    else:
        st.image('Images/Relation_fuel_cons_Ewltp_zoom_plateau.png')
    st.write('Nous voyons bien le lien linéaire fort entre les deux variables. Cependant, un plateau surprenant à 112g/km de rejet CO2 apparaît.')
    st.write('L\'objectif des prochains graphes est de trouver un dénominateur commun aux véhicules appartenant à ce plateau.')
    var = st.selectbox('Observer répartition de la variable :', ('Fuel_mode', 'Fuel_type', 'Manufacturer pooling', 'Country'))
    if var == 'Fuel_mode':
        st.image('Images/Repartition_plateau_fuel_mode.png')
    elif var == 'Fuel_type':
        st.image('Images/Repartition_plateau_fuel_type.png')
    elif var == 'Manufacturer pooling':
        st.image('Images/Repartition_plateau_manufacturer_pooling.png')
    else:
        st.image('Images/Repartition_plateau_country.png')
    st.write('Des répartitions ci-dessus, on retire que les véhicules sur le plateau sont principalement issus de Pologne. \
             Il n\'y a pas de raison particulière à cela, et la majorité de ces véhicules sont certainement renseignés en erreur.')
    st.write('En revenant sur le nuage de point, nous pouvons mettre en évidence la sur-représentationn des véhicules polonais sur le plateau.')
    pologne = st.checkbox('Mettre en évidence les véhicules de Pologne')
    if pologne == False:
        st.image('Images/Relation_fuel_cons_Ewltp_global.png')
    else:
        st.image('Images/Relation_fuel_cons_Ewltp_highlight_pologne.png')
    st.write('Finalement, nous enleverrons donc tous les véhicules polonais avec un rejet de CO2 à 112g/km.')
    st.code(f'''
            # Suppression des véhicules de Pologne avec un rejet de 112g/km
            df = df[(df['Country'] != 'PL') | (df['Ewltp_g_km'] != 112)]
            ''')
    st.write('#### Relation entre emission CO2 et la masse du véhicule')
    st.write('Nos variable de masse sont très peu corrélées au rejet CO2 si l\'on considère l\'ensemble de véhicules, \
             mais la corrélation est bien meilleure si l\'on sépare les types de motorisation thermique et hybride.\
             \nRegardons cela de plus près.')
    diff = st.checkbox('Séparer les véhicules thermiques et hybrides')
    if diff == False:
        st.image('Images/Relation_mass_Ewltp2.png')
    else:
        st.image('Images/Relation_mass_Ewltp_hybrid.png')
    st.write('La corrélation de la masse avec le rejet CO2 est bien important. \
             On note qu\'à masse égale, le niveau de rejet CO2 des véhicules hybrides est bien plus bas que celui de véhicules thermiques.')
    st.subheader('Variables Catégorielles')
    st.write('Cette partie se consacre aux variable catégorielles et à leur relations avec le rejet CO2')
    var = st.selectbox('Observer la relation de la variable :', ('Manufacturer pooling',
                                                                 'Fuel_type',
                                                                 'Fuel_mode',
                                                                 'Country',
                                                                 'Cat_vehcl_approved',
                                                                 'Cat_vehcl_registered'))
    if var == 'Fuel_mode':
        st.image('Images/Boxplot_Fuel_mode.png')
    elif var == 'Fuel_type':
        st.image('Images/Boxplot_Fuel_type.png')
    elif var == 'Manufacturer pooling':
        st.image('Images/Boxplot_Manufacturer pooling.png')
    elif var == 'Country':
        st.image('Images/Boxplot_Country.png')
    elif var == 'Cat_vehcl_approved':
        st.image('Images/Boxplot_Cat_vehcl_approved.png')
    else:
        st.image('Images/Boxplot_Cat_vehcl_registered.png')
    st.write('#### Relation entre emission CO2 et les type/mode de carburant')
    st.write('Les variables Fuel_type, et Fuel_mode par extension, sont celles qui nous intéressent le plus. Regardons leurs corrélations avec l\'émission CO2 de plus près.')
    st.image('Images/Boxplot_Fuel_type_Fuel_mode.png')
    st.write('Encore une fois, le grand écart de rejet CO2 entre thermique et hybride se démarque.')
    st.write('On peut aussi noter la faible importance du mode de carburant H. \
             En effet, les véhicules hybrides non rechargeables sont mesurés dans leurs mode de fonctionnement thermique, et donc aucun écart de rejet CO2 ne se fait sentir.')
    st.subheader('Choix des variables à considérer')
    st.write('Grâce à l\'ensemble de nos observations et analyses sur les données, nous pouvons choisir les variables que nous utiliserons pour nos modèles de machine learning.')
    st.write('Deux groupes de variables se distinguent alors :\
             \n* Variables indispensables : Fuel_type, Mass_wltp_kg, Engin_capcity_cm3, Horse_power_KW, Electric_range_km\
             \n* Variables optionnelles : Fuel_mode, Cat_vehcl_approved, Manufacturer_pooling, Fuel_cons, Electric_cons')

# Page 4
if page == 'Machine Learning':
    #def main():
        #st.set_page_config(page_title="Machine Learning", page_icon="🍀")
        #st.title("Analyse des résultats de Machine Learning")
    
    
    def regression_lineaire():
        Variable_options = st.selectbox("Options des Variables", [" ","Toutes les variables", "Variables indispensables"])
        Vehicule_options = st.selectbox("Type de véhicules", [" ", "Thermique", "Thermique et hybride"])
    
        # Vérifier si les options sont sélectionnées
        if Variable_options == " " or Vehicule_options == " ":
            st.write("Veuillez sélectionner les options des variables et le type de véhicules.")
            return
    
        image_paths = {
            ("Toutes les variables", "Thermique"): "FV_VTU.png",
            ("Toutes les variables", "Thermique et hybride"): "FV_VTH.png",
            ("Variables indispensables", "Thermique"): "VI_VTU.png",
            ("Variables indispensables", "Thermique et hybride"): "VI_VTH.png"
        }
        model_values = {
            "FV_VTU.png": {
                "LinearRegression": [63.108898, 7.944111, 4.953130, 0.959354],
                "XGBRegressor": [10.374093, 3.220884, 1.538607, 0.993318]
            },
            "FV_VTH.png": {
                "LinearRegression": [62.894203, 7.930587, 4.701509, 0.983899],
                "XGBRegressor": [14.105852, 3.755776, 1.583904, 0.996389]
            },
            "VI_VTU.png": {
                "LinearRegression": [ 300.713958, 17.341106, 12.992647, 0.815814],
                "XGBRegressor": [65.186753, 8.073831, 5.645208, 0.960073]
            },
            "VI_VTH.png": {
                "LinearRegression": [386.474785, 19.658962, 14.416041, 0.902178],
                "XGBRegressor": [93.915095, 9.690980, 5.361660, 0.976229]
            }
        }
        image_file = image_paths.get((Variable_options, Vehicule_options))
        image_path = os.path.join("Images/Regression", image_file)
        if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image = Image.open(image_path)
            st.image(image, caption=f"Image pour {Variable_options} - {Vehicule_options}")
        else:
            st.write("Aucune image trouvée ou le fichier n'est pas une image.")
        values = model_values.get(image_file, {})
        data = {
            "Model": ["LinearRegression", "XGBRegressor"],
            "Mean Squared Error": [values.get("LinearRegression", [0, 0, 0, 0])[0], values.get("XGBRegressor", [0, 0, 0, 0])[0]],
            "Root Mean Squared Error": [values.get("LinearRegression", [0, 0, 0, 0])[1], values.get("XGBRegressor", [0, 0, 0, 0])[1]],
            "Mean Absolute Error": [values.get("LinearRegression", [0, 0, 0, 0])[2], values.get("XGBRegressor", [0, 0, 0, 0])[2]],
            "R² Score": [values.get("LinearRegression", [0, 0, 0, 0])[3], values.get("XGBRegressor", [0, 0, 0, 0])[3]]
        }
        df = pd.DataFrame(data)
        st.dataframe(df)
        st.dataframe(df)
        st.markdown("<p style='text-align: justify;'>Nous avons 4 graphes ici, le premier nous présente les résultats de nos prédictions en comparant les valeurs réelles et les valeurs prédites et les 3 autres graphiques nous permettent de vérifier la présence d’un biais systématique en analysant le graphique des résidus, l’histogramme des résidus et le Q-Q plot.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Analyse des résultats de prédiction :</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Le graphique “Valeurs Réelles vs Valeurs Prédites” montre une forte corrélation entre les valeurs réelles et les valeurs prédites, ce qui indique que le modèle de régression linéaire fonctionne bien. La plupart des points sont alignés le long de la ligne de référence, ce qui signifie que les prédictions sont assez précises.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Valeurs Réelles vs Valeurs Prédites : Les points sont bien alignés le long de la ligne diagonale, montrant que les prédictions sont précises.Cela signifie que les erreurs de prédiction sont généralement faibles et réparties de manière aléatoire.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Cependant, il y a quelques points qui s’écartent de cette ligne, ce qui pourrait indiquer des outliers ou des cas où le modèle ne prédit pas aussi bien.</p>", unsafe_allow_html=True)
      
        st.markdown("<p style='text-align: justify;'>Vérification du biais Systématique :</p>", unsafe_allow_html=True)
    
        st.markdown("<p style='text-align: justify;'>Graphique des Résidus : Les résidus sont répartis de manière aléatoire autour de zéro, ce qui est un bon signe qu’il n’y a pas de biaise systématique.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Histogramme des Résidus : Les résidus semblent suivre une distribution normale.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Graphique Q-Q : Les points suivent approximativement la ligne diagonale, indiquant que les résidus sont normalement distribués.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Conclusion des modèles de régression :</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Ces graphiques confirment que Notre modèle de régression fonctionne bien et qu’il n’y a pas de biaise systématique apparente.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Choix du modèle</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Nous faisons le choix du XgBoost regrssor compte tenue de ses performaces comparé au autres modèles.</p>", unsafe_allow_html=True)
    
    def classification():
        st.markdown("<p style='text-align: justify;'>Pour préparer la mise en application en contexte, afin de répondre à des problématiques métiers, nous allons créer une variable taxe qui déterminera par un score la tranche de taxe qu'un constructeur devra payer compte tenu de ses rejets CO2.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Cette variable aura pour valeur 0, 1, 2, 3, correspondant à des niveaux de taxe CO2.</p>", unsafe_allow_html=True)
        
        # Créer un DataFrame avec les données de la taxe
        data = {
            "Score": [0, 1, 2, 3],
            "Description": ["Taxe de 0 €", "Taxe entre 0 € et 1000 €", "Taxe entre 1000 € et 10 000 €", "Taxe de plus de 10 000 €"]
        }
    
        df = pd.DataFrame(data)

        # Afficher le tableau avec st.table
        st.table(df)
    
        st.markdown("<p style='text-align: justify;'>Pour cette problématique de classification, nous allons utiliser le modèle Decision Tree Classifier et XGBoost Classifier.</p>", unsafe_allow_html=True)
    
        st.markdown("<p style='text-align: justify;'>Pour mesurer leur performance nous utiliserons la matrice de confusion et un Rapport de classification.</p>", unsafe_allow_html=True)
    
        st.markdown("<p style='text-align: justify;'>Ici il est primordial d’avoir de bonnes performances pour prédire la classe 3 puisque c’est cette partie qui contient la taxe la plus élevée. Si le modèle fait de mauvaise Prédiction sr cette classe des constructeurs seront taxés fortement alors qu’ils ne le devraient pas.</p>", unsafe_allow_html=True)
        
        Variable_options = st.selectbox("Options des Variables", [" ", "Toutes les variables", "Variables indispensables"])
        Vehicule_options = st.selectbox("Type de véhicules", [" ", "Thermiques", "Thermiques et hybrides"])
        Models_options = st.selectbox("Modeles", [" ", "Decision Tree Classifier", "XGBoost"])

        # Vérifier si les options sont sélectionnées
        if Variable_options == " " or Vehicule_options == " " or Models_options == " ":
            st.write("Veuillez sélectionner les options des variables, le type de véhicules et le modèle.")
            return

        image_paths = {
            ("Toutes les variables", "Thermiques", "Decision Tree Classifier"): ["FV_VT_MCDC.png", "FV_VT_RCDC.png"],
            ("Toutes les variables", "Thermiques et hybrides", "Decision Tree Classifier"): ["FV_VTH_MCDC.png", "FV_VTH_RCDC.png"],
            ("Variables indispensables", "Thermiques", "Decision Tree Classifier"): ["VI_VT_MCDC.png","VI_VT_RCDC.png"],
            ("Variables indispensables", "Thermiques et hybrides", "Decision Tree Classifier"): ["VI_VTH_MCDC.png", "VI_VTH_RCDC.png"],
            ("Toutes les variables", "Thermiques", "XGBoost"): ["FV_VT_MCXG.png", "FV_VT_RCXG.png"],
            ("Toutes les variables", "Thermiques et hybrides", "XGBoost"): ["FV_VTH_MCXG.png", "FV_VTH_RCXG.png"],
            ("Variables indispensables", "Thermiques", "XGBoost"): ["VI_VT_MCXG.png","VI_VT_RCXG.png"],
            ("Variables indispensables", "Thermiques et hybrides", "XGBoost"): ["VI_VTH_MCXG.png", "VI_VTH_RCXG.png"],
        }
        image_files = image_paths.get((Variable_options, Vehicule_options, Models_options))
        image_paths = [os.path.join("Images/Classification", image_file) for image_file in image_files]
        for image_path in image_paths:
            if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image = Image.open(image_path)
                st.image(image, caption=f"Image pour {Variable_options} - {Vehicule_options} - {Models_options}")
            else:
                st.write(f"Aucune image trouvée ou le fichier n'est pas une image: {image_path}")
                
        st.write("Globalement, le modèle semble bien fonctionner, avec un nombre élevé de prédictions correctes pour chaque classe.")
        # Vérifier si les options sont sélectionnées
        if Variable_options != " " and Vehicule_options != " " and Models_options != " ":
            # Créer un DataFrame avec les données fournies
            tableau_1 = {
                "Modèle": ["Decision Classifier Tree", "Decision Classifier Tree", "XGBoost", "XGBoost"],
                "Variables": ["Toutes les variables", "Toutes les variables", "Toutes les variables", "Toutes les variables"],
                "Type de Véhicules": ["Véhicules Thermiques", "Véhicules Thermiques et Hybrides", "Véhicules Thermiques", "Véhicules Thermiques et Hybrides"],
                "Precision": ["94 %", "97 %", "97 %", "98 %"],
                "Accuracy": ["96 %", "97 %", "97 %", "98 %"]
            }
    
            tableau_2 = {
                "Modèle": ["Decision Classifier Tree", "Decision Classifier Tree", "XGBoost", "XGBoost"],
                "Variables": ["Variable indispensable", "Variable indispensable", "Variables indispensable", "Variables indispensable"],
                "Type de Véhicules": ["Véhicules Thermiques", "Véhicules Thermiques et Hybrides", "Véhicules Thermiques", "Véhicules Thermiques et Hybrides"],
                "Precision": ["89 %", "88 %", "89 %", "89 %"],
                "Accuracy": ["89 %", "88 %", "89 %", "89 %"]
            }
            
            df_1 = pd.DataFrame(tableau_1)
            df_2 = pd.DataFrame(tableau_2)
    
            # Afficher les tableaux sans la colonne d'index
            st.table(df_1)
            st.table(df_2)
    
            st.write("Nous allons analyser deux métriques ici l’accuracy et la precision.")
            st.markdown("<p style='text-align: justify;'>L’Acuracy (indique la fréquence à laquelle un modèle ML de classification est correct dans l’ensemble).</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify;'>Nos classes n’étant pas équilibré, nous regarderons de plus prét ici la précision.</p>", unsafe_allow_html=True)
            
            st.markdown("<p style='text-align: justify;'>Précision( indique la fréquence à laquelle un modèle ML est correct lors de la prédiction de la classe cible).</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify;'>La précision est utile ici car le coût d’un faux positif est élevé. Si nous prédisons qu’un constructeur se situe dans la classe 0 alors qu’il est dans la classe 3, la facture risque d’être salée après coût.</p>", unsafe_allow_html=True)
    
            st.markdown("<p style='text-align: justify;'>Dans le premier tableau Nous analysons l’ensemble du dataset :</p>", unsafe_allow_html=True)
    
            st.markdown("<p style='text-align: justify;'>Pour le Decision Classifier Tree, le modèle est plus performant sur l’ensemble du dataset, mais le XGBoost obtient les mêmes performances avec uniquement les véhicules thermiques.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify;'>Dans le second tableau nous ne prenons que les variables indispensables.</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: justify;'>Ici les performances des deux modèles sont équivalentes, avec toutefois une toute petite perte de performance sur l’ensemble du dataset pour le Decision Tree Classifier, tandis que le XGBoost garde les mêmes performances sur les deux dataset.</p>", unsafe_allow_html=True)
    
            st.markdown("<p style='text-align: justify;'>Nous fessons donc le choix ici du XGBoost Classifier.</p>", unsafe_allow_html=True)
    
    def acp():
        st.markdown("<p style='text-align: justify;'>Dans l'optique d'optimiser nos modèles nous avons conçus une ACP avec dans un premier temps uniquement les véhicules thermiques ensuite nous avons inclus les véhicules hybrides..</p>", unsafe_allow_html=True)
        Vehicule_options = st.selectbox("Types de véhicules", [" ", "Thermiques", "Thermiques et hybrides"])
 
    # Vérifier si les options sont sélectionnées
        if Vehicule_options == " ":
            st.write("Veuillez sélectionner le type de véhicules.")
            return
 
        image_paths = {
        "Thermiques": ["VT_ACP_coude.png", "VT_PC1_PC2_brute.png", "VT_Cercle_correlation.png", "VT_Matrice_composant_PCA.png", "VT_PC1_PC2_traitement.png", "VT_Full_model_PCA.png", "Full_model.png"],
        "Thermiques et hybrides": ["VTH_ACP_coude.png", "VTH_PC1_PC2_brute.png", "VTH_Cercle_correlation.png", "VTH_Matrice_composant_PCA.png", "VTH_PC1_PC2_traitement.png", "VTH_Full_model_PCA.png", "Full_model.png"]
        }
        image_files = image_paths.get(Vehicule_options)
        if image_files is None:
            st.write("Aucune image trouvée pour l'option sélectionnée.")
            return
        image_files = image_paths.get(Vehicule_options)
        image_paths = [os.path.join("Images/ACP", image_file) for image_file in image_files]

        # Afficher les images en grille
        cols = st.columns(3)  # Nombre de colonnes dans la grille

        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image = Image.open(image_path)
                cols[i % 3].image(image, caption=f"Image pour {Vehicule_options}")
            else:
                cols[i % 3].write(f"Aucune image trouvée ou le fichier n'est pas une image: {image_path}")
 
        st.markdown("<p style='text-align: justify;'>En comparaison les modèles de régression sont plus précis avec une meilleure distribution des résidus autour de zéro.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Les histogrammes des résidus sont plus étalés en PCA sur les valeurs thermiques.</p>", unsafe_allow_html=True)
 
        st.markdown("<p style='text-align: justify;'>L’inclusion des véhicules hybrides permet d’améliorer les prédictions, observables avec la distribution des résidus, exceptées pour Elastic Net où c’est un peu plus dispersée.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Bien que les résultats ce soit améliorer, nous conservons les modèles de régression car ils sont plus précis.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Si on observe de plus près le XGBoost, on peut constater que sans l’analyse en PCA le XGBoost est plus efficace, la concentration des prédictions valeur réelle vs valeur prédite est moins dispersée.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify;'>Pour la mise en contexte nous voulons tester deux modèles pour vérifier quels modèles répondent le plus à notre problématique métier, alors nous écartons l’analyse en PCA qui est moins précise et retenons la régression linéaire ainsi que le XGBoost regressor.</p>", unsafe_allow_html=True)

    st.subheader('Machine Learning')
    st.markdown("""<p style='text-align: justify;'>L'Analyse des résultats de Machine learning va se faire en 3 parties, la première va être consacrée aux modèles de régressions, la deuxième partie sera consacrée à l'analyse des modèles de classifications et la dernière partie se fera sur l'analyse de l'ACP.</p>""", unsafe_allow_html=True)
    option = st.selectbox("Machine Learning", [" ", "Regression", "Classification", "ACP"])
    if option == "Regression":
        regression_lineaire()
    elif option == "Classification":
        classification()
    elif option == "ACP":
        acp()
       

# Page 5
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Load data
df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')

# Sidebar
with st.sidebar:
    st.title('🚗 Application en contexte')
    st.write('Cette partie propose d\'utiliser nos modèles de prédiction pour les appliquer à des exemples plus concrets.')
    st.write('Deux contextes d\'application sont proposés :\
             \n* **Constructeur automobile** :\
             \nDu point de vue d\'un constructeur automobile, on souhaite anticiper la taxe CO2 qui sera appliquée sur notre véhicule le plus tôt possible.\
             \n* **European Environment Agency** :\
             \nDu point de vue de l\'EEA, qui souhaite lorsqu\'un nouveau test WLTP est entré dans la base de données, vérifier la cohérence des données pour limiter les lignes mal renseignées.')
    contexte = st.selectbox('Choix du contexte', ['Constructeur auto', 'EEA'])

    if contexte == 'Constructeur auto':
        choix_model = st.selectbox('Choisir le modèle de classification',('DecisionTree','XGBClassifier'))
        if choix_model == 'DecisionTree':
            model_cla = pickle.load(open("models/model_DecisionTreeClassifier", 'rb'))
            transformer_cla = pickle.load(open("models/transformer_DecisionTreeClassifier.pkl", 'rb'))
        else:
            model_cla = pickle.load(open("models/model_XGBClassifier", 'rb'))
            transformer_cla = pickle.load(open("models/transformer_XGBClassifier.pkl", 'rb'))
        
        st.write('#### Définition du véhicule')
        Fuel_type = st.selectbox('Type de carburant', ('DIESEL', 'PETROL', 'LPG', 'NG', 'E85', 'HYBRID_D', 'HYBRID_P'))
        col1, col2 = st.columns([1,1])
        with col1:            
            Engin_power_KW = st.slider('Puissance (KW)', 50, 1200, 130, 10)   
            Mass_wltp_kg = st.slider('Masse du véhicule (kg)', 500, 5000, 1800, 10)
        with col2:
            Engin_capacity_cm3 = st.slider('Cylindré (cm3)', 600, 8000, 2000, 100)
            if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
                Electric_range_km = st.slider('Autonomie électrique (km)', 0, 1000, 50, 10)
            else:
                Electric_range_km = 0
        
        param = {'Mass_wltp_kg': [Mass_wltp_kg], 'Fuel_type': [Fuel_type], 'Engin_capcity_cm3':[Engin_capacity_cm3], 'Horse_power_KW':[Engin_power_KW], 'Electric_range_km':[Electric_range_km]}
        X = pd.DataFrame(data=param)
        X_transform = transformer_cla.transform(X)
        y_predict = model_cla.predict(X_transform)
        
        st.write('#### Taxe estimée')
        if y_predict[0] == 0:
            st.write(':green[Pas de taxe CO2 sur ce véhicule]')
        if y_predict[0] == 1:
            st.write(':blue[Taxe sur ce véhicule entre 0€ et 1.000€]')
        if y_predict[0] == 2:
            st.write(':orange[Taxe sur ce véhicule entre 1.000€ et 10.000€]')
        if y_predict[0] == 3:
            st.write(':red[Taxe sur ce véhicule au dessus de 10.000€]')
                
    else:
        choix_model = st.selectbox('Choisir le modèle de regression',('LinearRegression','XGBRegressor'))
        if choix_model == 'LinearRegression':
            model_reg = pickle.load(open("models/model_LinearRegression", 'rb'))
            transformer_reg = pickle.load(open("models/transformer_LinearRegression.pkl", 'rb'))
        else:
            model_reg = pickle.load(open("models/model_XGBRegressor", 'rb'))
            transformer_reg = pickle.load(open("models/transformer_XGBRegressor.pkl", 'rb'))
        
        st.write('#### Définition du véhicule')
        Exemple1 = st.button('Charger un exemple', key = 'Exemple1')
        Reset = st.button('Reset', key = 'Reset')
        afficher_illustration = False
        if (Exemple1 == True) | (afficher_illustration) == True:
            col1, col2 = st.columns([1,1])
            with col1:            
                Fuel_type = st.selectbox('Type de carburant', ('HYBRID_P', 'DIESEL', 'PETROL', 'LPG', 'NG', 'E85', 'HYBRID_D', 'HYBRID_P'), disabled=True)
                Fuel_mode = st.selectbox('Mode de carburant', ('P', 'M', 'H', 'B', 'F', 'P'), disabled=True)
            with col2:
                Cat_vehcl_approved = st.selectbox('Catégorie de véhicule', ('M1', 'M1', 'M1G', 'N1', 'N1G', 'N2', 'N2G'), disabled=True)
                Manufacturer_pooling = st.selectbox('Constructeur', ('VOLKSWAGEN', 'STELLANTIS', 'RENAULT-NISSAN-MITSUBISHI', 'SUBARU-SUZUKI-TOYOTA', 'HYUNDAI MOTOR EUROPE', 'KIA', 'MERCEDES-BENZ AG', 'FORD', 'VOLKSWAGEN', 'BMW', 'MAZDA', 'KG MOBILITY GREAT WALL MOTOR'), disabled=True)
            
            col_1, col_2 = st.columns([1,1])
            with col_1:
                col1, col2 = st.columns([2,1])
                with col2:
                    test1 = st.checkbox('Inconnu', key = 1, value = True, label_visibility= 'hidden')
                with col1:
                    Mass_wltp_kg = st.slider('Masse du véhicule', 500, 5000, 1777, 10, disabled=test1)
                
                col1, col2 = st.columns([2,1])
                with col2:
                    test2 = st.checkbox('Inconnu', key = 2, value = True, label_visibility= 'hidden')
                with col1:
                    Fuel_cons = st.slider('Consommation de carburant', 0.0, 25.0, 29.0, 0.1, disabled=test2)
                
                if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
                    col1, col2 = st.columns([2,1])
                    with col2:
                        test5 = st.checkbox('Inconnu', key = 5, value = True, label_visibility= 'hidden')
                    with col1:
                        Electric_range_km = st.slider('Autonomie électrique', 0, 1000, 57, 10, disabled=test5)
                else:
                    Electric_range_km = 0

            with col_2:
                col1, col2 = st.columns([2,1])
            with col2:
                test3 = st.checkbox('Inconnu', key = 3, value = True, label_visibility='hidden')
            with col1:
                Engin_capacity_cm3 = st.slider('Cylindré', 600, 8000, 1390, 100, disabled=test3)
    
            col1, col2 = st.columns([2,1])
            with col2:
                test4 = st.checkbox('Inconnu', key = 4, value = True, label_visibility='hidden')
            with col1:
                Engin_power_KW = st.slider('Puissance', 50, 1200, 110, 10, disabled=test4)
    
            if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
        
                col1, col2 = st.columns([2,1])
            with col2:
                test6 = st.checkbox('Inconnu', key = 6, value = True, label_visibility='hidden')
            with col1:
                Electric_cons = st.slider('Consommation électrique', 10, 600, 45, 10, disabled=test6)
            else:
                Electric_cons = 0

        if Fuel_type == 'INCONNU':
            Fuel_type = np.nan
        if Fuel_mode == 'INCONNU':
            Fuel_mode = np.nan
        if Cat_vehcl_approved == 'INCONNU':
            Cat_vehcl_approved = np.nan
        if Manufacturer_pooling == 'INCONNU':
            Manufacturer_pooling = np.nan

        st.write('#### Rejet CO2 mesuré')
        CO2 = st.slider('Rejet CO2', 0, 500, 120, 1)
        
        param = {'Mass_wltp_kg': [Mass_wltp_kg],
                    'Fuel_type': [Fuel_type],
                    'Engin_capcity_cm3':[Engin_capacity_cm3],
                    'Horse_power_KW':[Engin_power_KW],
                    'Electric_range_km':[Electric_range_km],
                    'Fuel_mode': [Fuel_mode],
                    'Cat_vehcl_approved': [Cat_vehcl_approved],
                    'Manufacturer_pooling': [Manufacturer_pooling],
                    'Fuel_cons': [Fuel_cons],
                    'Electric_cons': [Electric_cons],
                    }
        X = pd.DataFrame(data=param)
        X_transform = transformer_reg.transform(X)
        y_predict = model_reg.predict(X_transform)
        
        st.write('#### Evaluation de la cohérence des données')
        if round(abs((CO2-y_predict[0])/y_predict[0]*100),1) < 10:
            st.write(f':green[Pas d\'alerte.]  \nL\'émission CO2 renseignée diffère de {round(abs((CO2-y_predict[0])/y_predict[0]*100),1)}% de l\'émission attendue ({round(y_predict[0],0)})')
        elif round(abs((CO2-y_predict[0])/y_predict[0]*100),1) < 25:
            st.write(f':orange[Vérifiez les données, écart non négligeable.]  \nL\'émission CO2 renseignée diffère de {round(abs((CO2-y_predict[0])/y_predict[0]*100),1)}% de l\'émission attendue ({round(y_predict[0],0)})')
        else:
            st.write(f':red[Vérifiez les données, écart très important.]  \nL\'émission CO2 renseignée diffère de {round(abs((CO2-y_predict[0])/y_predict[0]*100),1)}% de l\'émission attendue ({round(y_predict[0],0)})')

        if Exemple1 == True:
            st.write('Graphiques explicatifs de la situation :')
            st.image('Images/Exemple_erreur_prediction.png')
            col1, col2, col3 = st.columns(3)            
            with col1:
                st.image('Images/ratio_petrol.png')
            with col2:
                st.image('Images/donnees_aberantes.png')
            with col3:
                st.image('Images/data_corrige.png')
                
if page == 'Dashboard':
    st.header('Application en contexte')
    st.write('Cette partie propose d\'utiliser nos modèles de prédiction pour les appliquer à des exemples plus concrets.')
    st.write('Deux contexte d\'application sont proposés :\
             \n* **Constructeur automobile** :\
             \nDu point de vue d\'un constructeur automobile, on souhaite anticiper la taxe CO2 qui sera appliquée sur notre véhicule le plus tôt pôssible.\
             \n* **European Environment Agency** :\
             \nDu point de vue de l\'EEA, qui souhaite lorsqu\'un nouveau test WLTP est entré dans la base de donnée, vérifier la cohérence des données pour limiter les lignes mal renseignées.')
    contexte = st.selectbox('Choix du contexte', ['Constructeur auto', 'EEA'])
    if contexte == 'Constructeur auto':
        st.write('  \n')
        choix_model = st.selectbox('Choisir le modèle de classification',('DecisionTree','XGBClassifier'))
        if choix_model == 'DecisionTree':
            model_cla = pickle.load(open("models/model_DecisionTreeClassifier", 'rb'))
            transformer_cla = pickle.load(open("models/transformer_DecisionTreeClassifier.pkl", 'rb'))
        else:
            model_cla = pickle.load(open("models/model_XGBClassifier", 'rb'))
            transformer_cla = pickle.load(open("models/transformer_XGBClassifier.pkl", 'rb'))
        st.write('  \n')
        st.write('#### Définition du véhicule')
        st.write('  \n')
        Fuel_type = st.selectbox('Type de carburant', ('DIESEL', 'PETROL', 'LPG', 'NG', 'E85', 'HYBRID_D', 'HYBRID_P'))
        col1, col2 = st.columns([1,1])
        with col1:            
            Engin_power_KW = st.slider('Puissance (KW)', 50, 1200, 130, 10)   
            
            Mass_wltp_kg = st.slider('Masse du véhicule (kg)', 500, 5000, 1800, 10)
        with col2:
            Engin_capacity_cm3 = st.slider('Cylindré (cm3)', 600, 8000, 2000, 100)
            if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
                Electric_range_km = st.slider('Autonomie électrique (km)', 0, 1000, 50, 10)
            else:
                Electric_range_km = 0
        param = {'Mass_wltp_kg': [Mass_wltp_kg], 'Fuel_type': [Fuel_type], 'Engin_capcity_cm3':[Engin_capacity_cm3], 'Horse_power_KW':[Engin_power_KW], 'Electric_range_km':[Electric_range_km]}
        X = pd.DataFrame(data=param)
        X_transform = transformer_cla.transform(X)
        y_predict = model_cla.predict(X_transform)
        st.write('  \n')
        st.write('#### Taxe estimée')
        if y_predict[0] == 0:
            st.write(':green[Pas de taxe CO2 sur ce véhicule]')
        if y_predict[0] == 1:
            st.write(':blue[Taxe sur ce véhicule entre 0€ et 1.000€]')
        if y_predict[0] == 2:
            st.write(':orange[Taxe sur ce véhicule entre 1.000€ et 10.000€]')
        if y_predict[0] == 3:
            st.write(':red[Taxe sur ce véhicule au dessus de 10.000€]')
                
    else:
        st.write('  \n')
        choix_model = st.selectbox('Choisir le modèle de regression',('LinearRegression','XGBRegressor'))
        if choix_model == 'LinearRegression':
            model_reg = pickle.load(open("models/model_LinearRegression", 'rb'))
            transformer_reg = pickle.load(open("models/transformer_LinearRegression.pkl", 'rb'))
        else:
            model_reg = pickle.load(open("models/model_XGBRegressor", 'rb'))
            transformer_reg = pickle.load(open("models/transformer_XGBRegressor.pkl", 'rb'))
        st.write('  \n')
        st.write('#### Définition du véhicule')
        Exemple1 = st.button('Charger un exemple', key = 'Exemple1')
        Reset = st.button('Reset', key = 'Reset')
        afficher_illustration = False
        if (Exemple1 == True) | (afficher_illustration) == True:
            st.write('  \n')
            col1, col2 = st.columns([1,1])
            with col1:            
                Fuel_type = st.selectbox('Type de carburant', ('HYBRID_P', 'DIESEL', 'PETROL', 'LPG', 'NG', 'E85', 'HYBRID_D', 'HYBRID_P'), disabled=True)
                Fuel_mode = st.selectbox('Mode de carburant', ('P', 'M', 'H', 'B', 'F', 'P'), disabled=True)
            with col2:
                Cat_vehcl_approved = st.selectbox('Catégorie de véhicule', ('M1', 'M1', 'M1G', 'N1', 'N1G', 'N2', 'N2G'), disabled=True)
                Manufacturer_pooling = st.selectbox('Constructeur', ('VOLKSWAGEN', 'STELLANTIS', 'RENAULT-NISSAN-MITSUBISHI', 'SUBARU-SUZUKI-TOYOTA', 'HYUNDAI MOTOR EUROPE', 'KIA', 'MERCEDES-BENZ AG', 'FORD', 'VOLKSWAGEN', 'BMW', 'MAZDA', 'KG MOBILITY GREAT WALL MOTOR'), disabled=True)
            st.write('  \n')
            col_1, col_2 = st.columns([1,1])
            with col_1:
                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test1 = st.checkbox('Inconnu', key = 1, value = True, label_visibility= 'hidden')
                with col1:
                    Mass_wltp_kg = st.slider('Masse du véhicule', 500, 5000, 1777, 10, disabled=test1)
                
                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test2 = st.checkbox('Inconnu', key = 2, value = True, label_visibility= 'hidden')
                with col1:
                    Fuel_cons = st.slider('Consommation de carburant', 0.0, 25.0, 29.0, 0.1, disabled=test2)
                

                if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
                    col1, col2 = st.columns([2,1])
                    with col2:
                        st.write('  \n')
                        st.write('  \n')
                        test5 = st.checkbox('Inconnu', key = 5, value = True, label_visibility= 'hidden')
                    with col1:
                        Electric_range_km = st.slider('Autonomie électrique', 0, 1000, 57, 10, disabled=test5)
                    
                else:
                    Electric_range_km = 0

            with col_2:
                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test3 = st.checkbox('Inconnu', key = 3, value = True, label_visibility= 'hidden')

                with col1:
                    Engin_capacity_cm3 = st.slider('Cylindré', 600, 8000, 1390, 100, disabled=test3)
                

                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test4 = st.checkbox('Inconnu', key = 4, value = True, label_visibility= 'hidden')
                with col1:
                    Engin_power_KW = st.slider('Puissance', 50, 1200, 110, 10, disabled=test4)
                
                if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
                    col1, col2 = st.columns([2,1])
                    with col2:
                        st.write('  \n')
                        st.write('  \n')
                        test6 = st.checkbox('Inconnu', key = 6, value = True, label_visibility= 'hidden')
                    with col1:
                        Electric_cons = st.slider('Consommation électrique', 10, 600, 45, 10, disabled=test6)
                    
                else:
                    Electric_cons = 0
        
            if Fuel_type == 'INCONNU':
                Fuel_type = np.nan
            if Fuel_mode == 'INCONNU':
                Fuel_mode = np.nan
            if Cat_vehcl_approved == 'INCONNU':
                Cat_vehcl_approved = np.nan
            if Manufacturer_pooling == 'INCONNU':
                Manufacturer_pooling = np.nan
            st.write('  \n')
            st.write('#### Rejet CO2 mesuré')
            st.write('  \n')
            CO2 = st.slider('Rejet CO2', 0, 500, 29, 1, disabled = True)
            
        
        elif (Exemple1 == False) | (Reset==True) :
            st.write('  \n')
            col1, col2 = st.columns([1,1])
            with col1:            
                Fuel_type = st.selectbox('Type de carburant', ('INCONNU', 'DIESEL', 'PETROL', 'LPG', 'NG', 'E85', 'HYBRID_D', 'HYBRID_P', 'INCONNU'))
                Fuel_mode = st.selectbox('Mode de carburant', ('INCONNU', 'M', 'H', 'B', 'F', 'P'))
            with col2:
                Cat_vehcl_approved = st.selectbox('Catégorie de véhicule', ('INCONNU', 'M1', 'M1G', 'N1', 'N1G', 'N2', 'N2G'))
                Manufacturer_pooling = st.selectbox('Constructeur', ('INCONNU', 'STELLANTIS', 'RENAULT-NISSAN-MITSUBISHI', 'SUBARU-SUZUKI-TOYOTA', 'HYUNDAI MOTOR EUROPE', 'KIA', 'MERCEDES-BENZ AG', 'FORD', 'VOLKSWAGEN', 'BMW', 'MAZDA', 'KG MOBILITY GREAT WALL MOTOR'))
            st.write('  \n')
            col_1, col_2 = st.columns([1,1])
            with col_1:
                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test1 = st.checkbox('Inconnu', key = 1)
                with col1:
                    Mass_wltp_kg = st.slider('Masse du véhicule', 500, 5000, 1800, 10, disabled=test1)
                if test1 == True:
                    Mass_wltp_kg = np.nan
                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test2 = st.checkbox('Inconnu', key = 2)
                with col1:
                    Fuel_cons = st.slider('Consommation de carburant', 0.0, 25.0, 8.0, 0.1, disabled=test2)
                if test2 == True:
                    Fuel_cons = np.nan

                if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
                    col1, col2 = st.columns([2,1])
                    with col2:
                        st.write('  \n')
                        st.write('  \n')
                        test5 = st.checkbox('Inconnu', key = 5)
                    with col1:
                        Electric_range_km = st.slider('Autonomie électrique', 0, 1000, 50, 10, disabled=test5)
                    if test5 == True:
                        Electric_range_km = np.nan
                else:
                    Electric_range_km = 0

            with col_2:
                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test3 = st.checkbox('Inconnu', key = 3)

                with col1:
                    Engin_capacity_cm3 = st.slider('Cylindré', 600, 8000, 2000, 100, disabled=test3)
                if test3 == True:
                    Engin_capacity_cm3 = np.nan

                col1, col2 = st.columns([2,1])
                with col2:
                    st.write('  \n')
                    st.write('  \n')
                    test4 = st.checkbox('Inconnu', key = 4)
                with col1:
                    Engin_power_KW = st.slider('Puissance', 50, 1200, 130, 10, disabled=test4)
                if test4 == True:
                    Engin_power_KW = np.nan
                if (Fuel_type == 'HYBRID_P') | (Fuel_type == 'HYBRID_D'):
                    col1, col2 = st.columns([2,1])
                    with col2:
                        st.write('  \n')
                        st.write('  \n')
                        test6 = st.checkbox('Inconnu', key = 6)
                    with col1:
                        Electric_cons = st.slider('Consommation électrique', 10, 600, 170, 10, disabled=test6)
                    if test6 == True:
                        Electric_cons = np.nan
                else:
                    Electric_cons = 0
        
            if Fuel_type == 'INCONNU':
                Fuel_type = np.nan
            if Fuel_mode == 'INCONNU':
                Fuel_mode = np.nan
            if Cat_vehcl_approved == 'INCONNU':
                Cat_vehcl_approved = np.nan
            if Manufacturer_pooling == 'INCONNU':
                Manufacturer_pooling = np.nan
            st.write('  \n')
            st.write('#### Rejet CO2 mesuré')
            st.write('  \n')
            CO2 = st.slider('Rejet CO2', 0, 500, 120, 1)
            
        param = {'Mass_wltp_kg': [Mass_wltp_kg],
                    'Fuel_type': [Fuel_type],
                    'Engin_capcity_cm3':[Engin_capacity_cm3],
                    'Horse_power_KW':[Engin_power_KW],
                    'Electric_range_km':[Electric_range_km],
                    'Fuel_mode': [Fuel_mode],
                    'Cat_vehcl_approved': [Cat_vehcl_approved],
                    'Manufacturer_pooling': [Manufacturer_pooling],
                    'Fuel_cons': [Fuel_cons],
                    'Electric_cons': [Electric_cons],
                    }
        X = pd.DataFrame(data=param)
        X_transform = transformer_reg.transform(X)
        y_predict = model_reg.predict(X_transform)
        st.write('  \n')
        st.write('#### Evaluation de la cohérence des données')
        st.write('  \n')
        if round(abs((CO2-y_predict[0])/y_predict[0]*100),1) < 10:
            st.write(f':green[Pas d\'alerte.]  \nL\'émission CO2 renseignée diffère de {round(abs((CO2-y_predict[0])/y_predict[0]*100),1)}% de l\'émission attendue ({round(y_predict[0],0)})')
        elif round(abs((CO2-y_predict[0])/y_predict[0]*100),1) < 25:
            st.write(f':orange[Vérifiez les données, écart non négligeable.]  \nL\'émission CO2 renseignée diffère de {round(abs((CO2-y_predict[0])/y_predict[0]*100),1)}% de l\'émission attendue ({round(y_predict[0],0)})')
        else:
            st.write(f':red[Vérifiez les données, écart très important.]  \nL\'émission CO2 renseignée diffère de {round(abs((CO2-y_predict[0])/y_predict[0]*100),1)}% de l\'émission attendue ({round(y_predict[0],0)})')

        if Exemple1 == True:
            st.write('  \n')
            st.write('Graphiques explicatifs de la situation :')
            st.image('Images/Exemple_erreur_prediction.png')
            col1, col2, col3 = st.columns(3)            
            with col1:
                st.image('Images/ratio_petrol.png')
            with col2:
                st.image('Images/donnees_aberantes.png')
            with col3:
                st.image('Images/data_corrige.png')


# Page 6
if page == 'Conclusion et perspectives':
    st.subheader('Conclusion et perspectives')    
    st.write('#### Machine Learning hors contexte')
    st.write('Nous avons pu entrainer et utiliser différents modèles de regression et de classification. \
    Les métriques obtenues avec ces modèles sont toutes très bonne, XGBoost se démarquant en tant que meilleur modèle.')
    st.write('#### Prédiction en contexte')
    st.write('Malgrés les bons résultats en terme de métrique, \
    l\'application du modèle XGBoost sur des cas concret montre que les métriques ne sont pas suffisantes pour juger de la validité d\'un modèle.  \
    \nEn l\'occurence, bien que XGBoost ait de meilleurs métriques, son application en contexte montre une forte sensibilité aux valeurs extrême/abérrantes.  \
    \nL\'utilisation de LinearRegression est alors à privilégier, car même si moins précise, cette régression moins sensible aux valeurs abérantes est plus adaptée au contexte métier.')
    st.write('#### Solutions envisageables')
    st.write('Pour pouvoir appliquer XGBoost en contexte, il faudrait au préalable \'nettoyer\' nos données par des méthodes plus poussées que celles utilisées. Par exemple :')
    st.write('- Regrouper les véhicules identiques mesurés plusieurs fois pour exclure les mesures abérrantes.')
    st.write('- Utiliser le ratio Rejet/Consommation pour exclure les véhicules aux ratios abérrants.')
    st.write('#### Enseignements principaux tirés du projet')
    st.write('- L\'analyse initiale des données doit être précise afin de bien cerner tous les enjeux du problème.')
    st.write('- Une des clé principale pour la réussite d\'un projet est de maîtriser le nettoyage des données.')
