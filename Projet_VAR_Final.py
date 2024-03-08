import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import yfinance as yf
import pylab as pl
from scipy.stats import norm
from arch import arch_model
import scipy.stats as stats
from scipy.stats import genextreme
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from matplotlib.backends.backend_pdf import FigureCanvasPdf
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.platypus import PageBreak
from reportlab.platypus import Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os
import pandas as pd
import matplotlib.dates as mdates
import scipy.stats as stats

#pip intall tkinter
#pip intall yfinance
#pip intall arch
#pip intall datetime
#pip intall scipy.stats
#pip intall matplotlib
#pip intall reportlab
#pip intall os
#pip intall pandas
#pip intall numpy



# Création de la fenêtre graphique
fenetre = tk.Tk()
fenetre.title("Calculateur de Value at Risk")
fenetre.geometry("1280x720")

# Création de variables globales
graphique_actuel = None
liste_VaR_Tab = []
liste_entreprises = ["Carrefour", "Vinci", "Danone", "Engie", "Kering SA", "Bouygues SA", "L'Oreal SA", "Schneider Electric S.E", "Sanofi", "L'Air Liquide S.A", "Atos SE", "Veolia Environnement SA"]
test = pd.DataFrame()
myDico = {"Carrefour": "CA.PA", "Vinci": "DG.PA", "Danone": "BN.PA", "Engie": "ENGI.PA", "Vivendi": "VIV.PA",
          "Kering SA": "KER.PA", "Bouygues SA": "EN.PA", "L'Oreal SA": "OR.PA", "Schneider Electric S.E": "SU.PA",
          "Sanofi": "SAN.PA", "L'Air Liquide S.A": "AI.PA", "Atos SE": "ATO.PA", "Veolia Environnement SA": "VIE.PA"}
flag = 0
# Fonction pour charger les données de toutes les entreprises
def charger_donnees_toutes_entreprises():
    df = pd.DataFrame()
    for nom, ticker in myDico.items():
        donnees = pl.array(yf.Ticker(ticker).history("3y").Close)
        df[nom] = donnees
    return df

# Fonction pour calculer les rendements
def rendements(df):
    X = np.zeros(np.shape(df))
    myData = pd.DataFrame(df)

    r = []
    if len(myData.columns) > 1:
        for j in range(len(myData.columns)):
            for i in range(len(df) - 1):
                X[i][j] = np.log(df.iloc[i + 1, j]) - np.log(df.iloc[i, j])
        return pd.DataFrame(X)
    else:
        r = []
        for i in range(1, len(df)):
            prix_actuel = df.iloc[i]
            prix_precedent = df.iloc[i - 1]
            rendement = (prix_actuel - prix_precedent) / prix_precedent
            r.append(rendement)
        return pd.DataFrame(r)



# Fonction pour calculer la VaR historique
def VAR_historique(df, X, h, confidence_level, nbj):
    if len(X) > 1:
        portfolio_returns = np.dot(X,h)
    else:
        X = rendements(df)
        p = 1
        portfolio_returns = np.dot(X, p)

    # Calcul de la VaR historique
    var_portfolio = np.sqrt(nbj) * np.percentile(portfolio_returns, 100 - confidence_level * 100)

    return round(var_portfolio,6)

# Fonction pour calculer la VaR paramétrique
def VAR_parametrique(df, X, w, confidence_level, nbj):
    
    cov = np.cov(X, rowvar=False) * nbj
    portfolio_returns = np.dot(X, w)
    moyenne = np.mean(portfolio_returns) * nbj
    VaR_param = -(moyenne + np.sqrt(np.dot(w, np.dot(cov, w))) * norm.ppf(confidence_level))
    
    return round(VaR_param,6)

# Fonction pour calculer la VaR Garch
def VAR_Garch(df, X, w, confidence_level, nbj):

    portfolio_returns = np.dot(X, w)

    garch_model = arch_model(portfolio_returns, vol = "Garch", p = 1, q = 1)
    resultat = garch_model.fit()

    omega = resultat.params[1]
    alpha = resultat.params[2]
    beta = resultat.params[3]

    sigma2 = omega / (1 - alpha - beta) #Variance inconditionnelle

    prevision = resultat.forecast(horizon = 1)
    h_th = prevision.variance[-1:].values[0, 0]

    var_garch = - np.sqrt(nbj * sigma2 + (h_th - sigma2) * ((1 - (alpha + beta)**nbj)/(1 - alpha - beta))) * norm.ppf(confidence_level)
    
    return round(var_garch,6)

# Fonction pour calculer la VaR Cornish-Fisher
def VAR_CF(df, X, w, confidence_level, nbj):
    
    portfolio_returns = np.dot(X, w)

    garch_model = arch_model(portfolio_returns, vol = "Garch", p = 1, q = 1)
    resultat = garch_model.fit()

    omega = resultat.params[1]
    alpha = resultat.params[2]
    beta = resultat.params[3]
    sigma2 = omega / (1 - alpha - beta) #Variance inconditionnelle

    S = stats.skew(portfolio_returns)
    K = stats.kurtosis(portfolio_returns)

    a = norm.ppf(confidence_level)

    k_inv = a + ((a**2 - 1) / 6) * S + ((a**3 - 3 * a) / 24) * K + ((2 * a**3 - 5 * a) / 36) * S**2

    prevision = resultat.forecast(horizon = 1)
    h_th = prevision.variance[-1:].values[0, 0]


    var_CF = - np.sqrt(nbj * sigma2 + (h_th - sigma2) * ((1 - (alpha + beta)**nbj)/(1 - alpha - beta))) * k_inv
    
    return round(var_CF,6)


def VAR_RM(df, X, w, confidence_level, nbj):
    
    Mat_var = np.zeros(len(w))
    for i in range(len(w)):
        T = X[i]**2 * w[i]

        Z = T.ewm(alpha = 0.30, adjust = False).mean()
        Z_vol = np.sqrt(Z)
        Vol_cond = Z_vol.iloc[-1]
        var = Vol_cond * norm.ppf(confidence_level)
        Mat_var[i] = var


    corr = X.corr()
    var_RM = - np.sqrt(np.dot(np.dot(Mat_var,corr),Mat_var)) 
    
    return round(var_RM,6)


def VAR_TVE(df, X, w, confidence_level, nbj):
    
    portfolio_returns = np.dot(X, w)
    threshold = np.percentile(portfolio_returns, confidence_level)

    excess_returns = portfolio_returns[portfolio_returns > threshold] - threshold
    
    params = stats.genpareto.fit(excess_returns)

    VaR = threshold + stats.genpareto.ppf(1 - confidence_level, *params)

    return round(VaR,6)

def VAR_TVE_Garch(df, X, w, confidence_level, nbj):
    
    VaR_Garch = VAR_Garch(df, X, w, confidence_level, nbj)
    
    portfolio_returns = np.dot(X, w)
    params = genextreme.fit(portfolio_returns)
    var_TVE = genextreme.ppf(confidence_level, *params) * nbj ** params[2]

    var_TVE_Garch = (VaR_Garch + var_TVE) / 2
    
    if var_TVE_Garch < 0:
        return round(var_TVE_Garch,6)
    else:
        return round(-var_TVE_Garch,6)

# Fonction pour afficher les données dans l'onglet "Base de données"
def afficher_donnees_dans_onglet_base_de_donnees(df):
    onglet_donnees_text.delete("1.0", tk.END)
    onglet_donnees_text.insert(tk.END, df.to_string())


label_text_onglet1 = tk.StringVar()
label_text_onglet2 = tk.StringVar()
label_text_onglet3 = tk.StringVar()
label_text_onglet4 = tk.StringVar()
label_text_onglet5 = tk.StringVar()
label_text_onglet6 = tk.StringVar()
label_text_onglet7 = tk.StringVar()

# Fonction pour placer les labels dans les graphiques
def placer_labels(subplot, onglet_index, nom):
    label_text = ""
    if onglet_index == 0:
        label_text = label_text_onglet1.get()
    elif onglet_index == 1:
        label_text = label_text_onglet2.get()
    elif onglet_index == 2:
        label_text = label_text_onglet3.get()
    elif onglet_index == 3:
        label_text = label_text_onglet4.get()
    elif onglet_index == 4:
        label_text = label_text_onglet5.get()
    elif onglet_index == 5:
        label_text = label_text_onglet6.get()
    elif onglet_index == 6:
        label_text = label_text_onglet7.get()
    liste_VaR_Tab.append(label_text)
    label_var1 = subplot.text(1., 0.2, label_text, transform=subplot.transAxes, va='center', ha='left', fontsize=12, fontweight='bold')
    label_var2 = subplot.text(1., 0.3, nom[onglet_index], transform=subplot.transAxes, va='center', ha='left', fontsize=12, fontweight='bold')

# Fonction pour afficher une boîte de dialogue demandant la proportion de chaque entreprise sélectionnée
def demander_proportion(entreprise):
    proportion = simpledialog.askfloat("Proportion", f"Entrez la proportion pour l'entreprise {entreprise} :", parent=fenetre)
    return proportion

# Fonction pour afficher une boîte de dialogue demandant le niveau de confiance
def demander_Niveau_Confiance():
    niveau_confiance = simpledialog.askfloat("Taux de couverture", "Entrez le taux de couverture :", parent=fenetre)
    return niveau_confiance

# Fonction pour afficher une boîte de dialogue demandant le nombre de jours
def demander_Nombre_jours():
    nbj = simpledialog.askfloat("Horizon du risque", "Entrez l'horizon du risque :", parent=fenetre)
    return nbj


def ajouter_tableau(dates_datetime):
    
    onglet_tableau = ttk.Frame(onglets)
    onglets.add(onglet_tableau, text="Statistiques")
    
    
    df_rend = rendements(donnees_base_de_donnees)
    pf_returns = np.dot(df_rend, h)
    
    tree = ttk.Treeview(onglet_tableau)

    # Ajouter des colonnes au tableau
    tree["columns"] = ("Performances", "Résultat")
    
    tree.column("#0", width=0, stretch=tk.NO)  # Colonne invisible
    tree.column("Performances", anchor=tk.CENTER, width=100)
    tree.column("Résultat", anchor=tk.CENTER, width=100)
    
    # Définir les en-têtes de colonne
    tree.heading("#0", text="", anchor=tk.CENTER)
    tree.heading("Performances", text="Performances", anchor=tk.CENTER)
    tree.heading("Résultat", text="Résultat", anchor=tk.CENTER)

    
    tree.insert("", tk.END, values=("Moyenne", round(np.mean(pf_returns),6)))
    tree.insert("", tk.END, values=("Variance", round(np.var(pf_returns),6)))
    tree.insert("", tk.END, values=("Ecart-type", round(np.sqrt(np.var(pf_returns)),6)))
    tree.insert("", tk.END, values=("Médiane", round(np.median(pf_returns), 6)))
    tree.insert("", tk.END, values=("1er quantile", round(np.quantile(pf_returns, 0.25),6)))
    tree.insert("", tk.END, values=("3eme quantile", round(np.quantile(pf_returns, 0.75),6)))
    tree.insert("", tk.END, values=("4eme quantile", round(np.quantile(pf_returns, 1),6)))
    tree.insert("", tk.END, values=("Skewness", round(stats.skew(pf_returns),6)))
    tree.insert("", tk.END, values=("Kurtosis", round(stats.kurtosis(pf_returns),6)))
    
    # Afficher le tableau
    tree.place(x = 550, y = 150)
    
    # Créer une fenêtre
    onglet_tableau = ttk.Frame(onglets)
    onglets.add(onglet_tableau, text="Résumé VaR graphique")
    
    figure = Figure(figsize=(8, 6), dpi=70)
    subplot3 = figure.add_subplot(111)

    subplot3.plot(pf_returns)
    subplot3.set_title("Rendements du portefeuille")
    subplot3.set_xlabel("Dates")
    subplot3.set_ylabel("Rendements")
    subplot3.axhline(y= float(liste_VaR_Tab[0]), color='r', linestyle='--', label = "VaR historique", linewidth=2)
    subplot3.axhline(y= float(liste_VaR_Tab[1]), color='blue', linestyle='--', label = "VaR paramétrique", linewidth=2)
    subplot3.axhline(y= float(liste_VaR_Tab[2]), color='brown', linestyle='--', label = "VaR Garch", linewidth=2)
    subplot3.axhline(y= float(liste_VaR_Tab[3]), color='green', linestyle='--', label = "VaR Cornish-Fischer", linewidth=2)
    subplot3.axhline(y= float(liste_VaR_Tab[4]), color='black', linestyle='--', label = "VaR Riskmetrics", linewidth=2)
    subplot3.axhline(y= float(liste_VaR_Tab[5]), color='orange', linestyle='--', label = "VaR TVE", linewidth=2)
    subplot3.axhline(y= float(liste_VaR_Tab[6]), color='grey', linestyle='--', label = "VaR TVE-Garch", linewidth=2)
    subplot3.legend(loc = "upper right")
    subplot3.set_xticks(np.arange(len(pf_returns))[::180])
    subplot3.set_xticklabels(dates_datetime[::180])
    canvas = FigureCanvasTkAgg(figure, master=onglet_tableau)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    graphique_actuel = canvas
    liste_graph.append(figure)
    
    
liste_graph = []
def envoyer():
    global graphique_actuel
    global myDico
    global checkbuttons_gauche
    global checkbuttons_droite
    global donnees_base_de_donnees
    global h
    
    entreprises_selectionnees_gauche = [entreprise for entreprise, var in checkbuttons_gauche if var.get() == 1]
    entreprises_selectionnees_droite = [entreprise for entreprise, var in checkbuttons_droite if var.get() == 1]
    
    entreprises_selectionnees = entreprises_selectionnees_gauche + entreprises_selectionnees_droite

    if not entreprises_selectionnees:
        label_resultat.config(text="Erreur: Veuillez sélectionner au moins une entreprise.")
        return

    # Création d'un dictionnaire pour stocker les données des entreprises et les proportions
    donnees_entreprises = {}
    p = {}
    invalid_proportions = []  # Liste pour stocker les entreprises avec des proportions invalides
    h = []
    for entreprise in liste_entreprises:
        p[entreprise] = 0  # Initialisation de la proportion à 0 pour toutes les entreprises non sélectionnées
        if entreprise in entreprises_selectionnees:
            # Demander la proportion pour l'entreprise sélectionnée
            proportion = demander_proportion(entreprise)
            if proportion is not None:
                p[entreprise] = proportion
                h.append(proportion)
                nom = myDico.get(entreprise)
                if nom:
                    donnees = pl.array(yf.Ticker(nom).history("3y").Close)
                    donnees_entreprises[entreprise] = donnees
            else:
                invalid_proportions.append(entreprise)
    
    ind0 = pd.DataFrame(yf.Ticker("CA.PA").history("3y").Close)
    dates = ind0.index.strftime('%d/%m/%Y')
    dates_datetime = pd.to_datetime(dates, format="%d/%m/%Y")
    
    # Vérification que la somme des proportions est égale à 1
    sum_proportions = sum(p.values())
    if sum_proportions != 1:
        messagebox.showerror("Erreur", "La somme des proportions doit être égale à 1.", parent=fenetre)
        return

    # Affichage d'un messagebox unique pour les entreprises avec des proportions invalides
    if invalid_proportions:
        messagebox.showerror("Erreur", f"Les entreprises suivantes ont des proportions invalides : {', '.join(invalid_proportions)}", parent=fenetre)
        return
    
    global confidence_level
    confidence_level = demander_Niveau_Confiance()
    nbj = demander_Nombre_jours()
    # Création d'un DataFrame à partir du dictionnaire de données
    df = pd.DataFrame(donnees_entreprises)
    X = rendements(df)
    
    # Stockage des données dans le DataFrame global
    donnees_base_de_donnees = df

    # Affichage des données dans l'onglet "Base de données"
    afficher_donnees_dans_onglet_base_de_donnees(df)

    # Modification des valeurs des labels
    label_text_onglet1.set(VAR_historique(df, X, h, confidence_level, nbj))
    label_text_onglet2.set(VAR_parametrique(df, X, h, confidence_level, nbj))
    label_text_onglet3.set(VAR_Garch(df, X, h, confidence_level, nbj))
    label_text_onglet4.set(VAR_CF(df, X, h, confidence_level, nbj))
    label_text_onglet5.set(VAR_RM(df, X, h, confidence_level, nbj)) 
    label_text_onglet6.set(VAR_TVE(df, X, h, confidence_level, nbj))
    label_text_onglet7.set(VAR_TVE_Garch(df, X, h, confidence_level, nbj))

    
    # Affichage des graphiques dans les onglets "VaR Historique" et "VaR Paramétrique"
    for i, frame in enumerate(frame_onglets[:-1]):
        for widget in frame.winfo_children():
            widget.destroy()

        if graphique_actuel and i == 0:
            graphique_actuel.get_tk_widget().destroy()

        figure = Figure(figsize=(8, 6), dpi=70)


        subplot2 = figure.add_subplot(111)
        df_rend = rendements(donnees_base_de_donnees)
        pf_returns = np.dot(df_rend, h)
        
        figure.subplots_adjust(hspace=0.5)
        placer_labels(subplot2, i, noms_onglets)
        
        subplot2.plot(pf_returns)
        subplot2.axhline(y= float(liste_VaR_Tab[i]), color='r', linestyle='--', label = noms_onglets[i])

            
        subplot2.set_title("Rendements du portefeuille")
        subplot2.set_xlabel("Dates")
        subplot2.set_xticks(np.arange(len(pf_returns))[::180])
        subplot2.set_xticklabels(dates_datetime[::180])
        subplot2.set_ylabel("Rendements")
        subplot2.legend(loc = "upper right")
        
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        graphique_actuel = canvas
        liste_graph.append(figure)
    ajouter_tableau(dates_datetime)
    
# Création des onglets avec les noms modifiés
onglets = ttk.Notebook(fenetre)
frame_onglets = []
noms_onglets = ["VaR Historique", "VaR Paramétrique", "VaR Garch", "VaR Riskmetrics", "VaR Cornish-Fischer", "VaR TVE", "VaR TVE-GARCH","Base de données"]

test_dico = {}
for nom in noms_onglets:
    test_dico[nom] = []
    
def pre_backtesting(VAR, rendement, alpha):
    
    #Matrice des violations
    x = alpha
    alpha = 1 - x
    T = len(rendement)

    log_returns = np.log(1 + rendement)

    HIT = np.zeros(T-1) 
    
    for i in range(1, T):
        if log_returns[i] < float(VAR):
            HIT[i-1] = 1
            
    nb_violations = np.sum(HIT)
    
    corr_matrix = np.corrcoef(HIT[:-1], HIT[1:])
    rho = corr_matrix[0, 1]
    
    LRcc = T * (np.log(1 - rho**2) - np.log(1 - (rho * (1 + rho)) / (1 - rho)))

    Z1 = (nb_violations - alpha * T) / np.sqrt(alpha * (1 - alpha) * T) #Test de Kupiec I   
    Z2 = -2 * np.log(1 + (1 - alpha)**(T - nb_violations) * alpha**nb_violations) + 2 * np.log((1 - (nb_violations / T))**(T-nb_violations) * (nb_violations / T)**nb_violations) #Test de Kupiec II
    
    return nb_violations, round(LRcc,6), round(Z1,6), round(Z2,6)         
            

for nom_onglet in noms_onglets:
    frame_onglet = ttk.Frame(onglets)
    onglets.add(frame_onglet, text=nom_onglet)
    frame_onglets.append(frame_onglet)

onglets.pack(expand=1, fill="both")

# Diviser la liste des entreprises en deux groupes
nb_entreprises = len(liste_entreprises)
midpoint = nb_entreprises // 2
liste_entreprises_gauche = liste_entreprises[:midpoint]
liste_entreprises_droite = liste_entreprises[midpoint:]

# Création des cadres pour les widgets Checkbutton
frame_entreprises_gauche = tk.Frame(fenetre)
frame_entreprises_droite = tk.Frame(fenetre)
frame_entreprises_gauche.pack(side="left", padx=10)
frame_entreprises_droite.pack(side="right", padx=10)

frame_entreprises = tk.Frame(fenetre)
frame_entreprises.pack()

frame_entreprises_gauche = tk.Frame(frame_entreprises)
frame_entreprises_gauche.pack(side="left", padx=10)

frame_entreprises_droite = tk.Frame(frame_entreprises)
frame_entreprises_droite.pack(side="left", padx=10)

checkbuttons_gauche = []
for entreprise in liste_entreprises_gauche:
    var = tk.IntVar()
    checkbutton = tk.Checkbutton(frame_entreprises_gauche, text=entreprise, variable=var)
    checkbutton.pack(anchor='w')
    checkbuttons_gauche.append((entreprise, var))

checkbuttons_droite = []
for entreprise in liste_entreprises_droite:
    var = tk.IntVar()
    checkbutton = tk.Checkbutton(frame_entreprises_droite, text=entreprise, variable=var)
    checkbutton.pack(anchor='w')
    checkbuttons_droite.append((entreprise, var))

onglet_donnees_text = tk.Text(frame_onglets[-1])
onglet_donnees_text.pack(fill="both", expand=True)

bouton_envoyer = tk.Button(fenetre, text="Envoyer", command=envoyer)
bouton_envoyer.pack(pady=10)


def reporting():
    
    styles = getSampleStyleSheet()
    # Définir un nouveau style pour le sous-titre
    subtitle_style = ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontSize=12, leading=16)
    # Ajouter le nouveau style au jeu de styles
    styles.add(subtitle_style)
    
    if liste_graph == []:
        messagebox.showerror("Erreur", "Le reporting ne peut pas être fait", parent=fenetre)
        return 
    
    if flag == 0:
        messagebox.showerror("Erreur", "Le backtesting doit être fait avant de faire le reporting", parent=fenetre)
        return
    
    df_rend = rendements(donnees_base_de_donnees)
    pf_returns = np.dot(df_rend, h)
    
    graphe_par_page = 2
    nombre_pages = int(np.ceil(len(liste_graph) / graphe_par_page))

    # Créer un nouveau fichier PDF
    pdf_filename = "reporting.pdf"
    pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)
    
    # Ajouter du texte au PDF
    elements = []
    elements.append(Paragraph("Reporting des calculs de Value at Risk", styles['Title']))
    
    data =[ 
    ["Moyenne", "Variance", "Ecart-type", "Médiane", "1er quantile", "3eme quantile", "4eme quantile", "Skewness", "Kurtosis"],
    [round(np.mean(pf_returns),6), round(np.var(pf_returns),6), round(np.sqrt(np.var(pf_returns)),6), round(np.median(pf_returns), 6), round(np.quantile(pf_returns, 0.25),6), round(np.quantile(pf_returns, 0.75),6), round(np.quantile(pf_returns, 1),6), round(stats.skew(pf_returns),6), round(stats.kurtosis(pf_returns),6)],
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(Paragraph("Tableau des performances :", styles['Subtitle']))
    elements.append(Spacer(1, 12))
    elements.append(table)
    
    z = 0
    for page in range(nombre_pages):
        if page != 0:
            elements.append(PageBreak())
            
        for idx in range(graphe_par_page):
            graphique_idx = page * graphe_par_page + idx
            if graphique_idx < len(liste_graph):
                # Dessiner le graphique sur le canvas
                fig = liste_graph[graphique_idx]
                img_path = f"temp_graph_{graphique_idx}.png"
                fig.savefig(img_path)
                plt.close(fig)  # Fermer la figure pour libérer la mémoire
                if z == 7:
                    elements.append(Paragraph(f"Graphique {graphique_idx + 1} : Aggrégation", styles['Subtitle']))
                else:
                    elements.append(Paragraph(f"Graphique {graphique_idx + 1} : {noms_onglets[z]}", styles['Subtitle']))
                z = z + 1
                elements.append(Spacer(1, 36))
                elements.append(Image(img_path, width=500, height=300))
    
    # Ajouter le tableau
    data = [
        ["VaR historique", "VaR paramétrique", "VaR Garch", "VaR Cornish-Fischer", "VaR Riskmetrics", "VaR TVE", "Var TVE-Garch"],
        [float(liste_VaR_Tab[0]), float(liste_VaR_Tab[1]), float(liste_VaR_Tab[2]), float(liste_VaR_Tab[3]), float(liste_VaR_Tab[4]), float(liste_VaR_Tab[5]), float(liste_VaR_Tab[6])],
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    # elements.append(table)
    elements.append(Paragraph("Tableau des Valeurs at Risk :", styles['Subtitle']))
    elements.append(Spacer(1, 12))
    elements.append(table)
    
    
    # Données pour le tableau
    
    list_histo1 = pre_backtesting(liste_VaR_Tab[0], pf_returns, confidence_level)
    list_histo2 = pre_backtesting(liste_VaR_Tab[1], pf_returns, confidence_level)
    list_histo3 = pre_backtesting(liste_VaR_Tab[2], pf_returns, confidence_level)
    list_histo4 = pre_backtesting(liste_VaR_Tab[3], pf_returns, confidence_level)
    list_histo5 = pre_backtesting(liste_VaR_Tab[4], pf_returns, confidence_level)
    list_histo6 = pre_backtesting(liste_VaR_Tab[5], pf_returns, confidence_level)
    list_histo7 = pre_backtesting(liste_VaR_Tab[6], pf_returns, confidence_level)
    # Données pour le tableau (à remplacer par vos propres données)
    data = [
    ["VaR", "Violations", "Christoffersen", "Kupiec I", "Kupiec II"],
    [liste_VaR_Tab[0], list_histo1[0], list_histo1[1], list_histo1[2], list_histo1[3]],
    [liste_VaR_Tab[1], list_histo2[0], list_histo2[1], list_histo2[2], list_histo2[3]],
    [liste_VaR_Tab[2], list_histo3[0], list_histo3[1], list_histo3[2], list_histo3[3]],
    [liste_VaR_Tab[3], list_histo4[0], list_histo4[1], list_histo4[2], list_histo4[3]],
    [liste_VaR_Tab[4], list_histo5[0], list_histo5[1], list_histo5[2], list_histo5[3]],
    [liste_VaR_Tab[5], list_histo6[0], list_histo6[1], list_histo6[2], list_histo6[3]],
    [liste_VaR_Tab[6], list_histo7[0], list_histo7[1], list_histo7[2], list_histo7[3]]
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    # elements.append(table)
    elements.append(Paragraph("Tableau de Backtesting :", styles['Subtitle']))
    elements.append(Spacer(1, 12))
    elements.append(table)
    
    # # Écrire les éléments dans le PDF
    pdf.build(elements)
    
    png_files = [f"temp_graph_{idx}.png" for idx in range(len(liste_graph))]

    # Supprimer chaque fichier PNG
    for file in png_files:
        if os.path.exists(file):
            os.remove(file)
    
    
    pdf_path = "reporting.pdf"

    # Vérifier si le fichier PDF existe
    if os.path.exists(pdf_path):
    # Afficher un message avec l'emplacement du fichier PDF
        messagebox.showinfo("Fichier PDF", f"Le fichier PDF a été créé avec succès à l'emplacement :\n{os.path.abspath(pdf_path)}")
    

def backtesting():
    
    if liste_graph == []:
        messagebox.showerror("Erreur", "Le backtesting ne peut pas être fait", parent=fenetre)
        return
    
    global flag
    flag = 1
    #Tableau backtesting
    onglet_tableau = ttk.Frame(onglets)
    onglets.add(onglet_tableau, text="Backtesting")

    #Tableau Value at Risk
    # Créer un Treeview
    tree = ttk.Treeview(onglet_tableau)

    # Ajouter des colonnes au tableau
    tree["columns"] = ("VaR", "Résultat")
    
    tree.column("#0", width=0, stretch=tk.NO)  # Colonne invisible
    tree.column("VaR", anchor=tk.CENTER, width=100)
    tree.column("Résultat", anchor=tk.CENTER, width=100)
    
    # Définir les en-têtes de colonne
    tree.heading("#0", text="", anchor=tk.CENTER)
    tree.heading("VaR", text="VaR", anchor=tk.CENTER)
    tree.heading("Résultat", text="Résultat", anchor=tk.CENTER)

    
    tree.insert("", tk.END, values=("VaR historique", liste_VaR_Tab[0]))
    tree.insert("", tk.END, values=("VaR paramétrique", liste_VaR_Tab[1]))
    tree.insert("", tk.END, values=("VaR Garch", liste_VaR_Tab[2]))
    tree.insert("", tk.END, values=("VaR Cornish-Fischer", liste_VaR_Tab[3]))
    tree.insert("", tk.END, values=("VaR Riskmetrics", liste_VaR_Tab[4]))
    tree.insert("", tk.END, values=("VaR TVE", liste_VaR_Tab[5]))
    tree.insert("", tk.END, values=("VaR TVE-Garch", liste_VaR_Tab[6]))
    
    # Afficher le tableau
    tree.place(x = 550, y = 10)
    
    #Tableau Value at Risk
    # Créer un Treeview
    tree1 = ttk.Treeview(onglet_tableau)

    # Ajouter des colonnes au tableau
    tree1["columns"] = ("VaR", "Violations", "Christoffersen", "Kupiec I", "Kupiec II")
    

    # Définir le nom des colonnes
    tree1.column("#0", width=0, stretch=tk.NO)  # Colonne invisible
    tree1.column("VaR", anchor=tk.CENTER, width=100)
    tree1.column("Violations", anchor=tk.CENTER, width=100)
    tree1.column("Christoffersen", anchor=tk.CENTER, width=100)
    tree1.column("Kupiec I", anchor=tk.CENTER, width=100)
    tree1.column("Kupiec II", anchor=tk.CENTER, width=100)
    
    # Définir les en-têtes de colonne
    tree1.heading("#0", text="", anchor=tk.CENTER)
    tree1.heading("VaR", text="VaR", anchor=tk.CENTER)
    tree1.heading("Violations", text="Violations", anchor=tk.CENTER)
    tree1.heading("Christoffersen", text="Christoffersen", anchor=tk.CENTER)
    tree1.heading("Kupiec I", text="Kupiec I", anchor=tk.CENTER)
    tree1.heading("Kupiec II", text="Kupiec II", anchor=tk.CENTER)
    

    # Ajouter des données au tableau
    df_rend = rendements(donnees_base_de_donnees)
    pf_returns = np.dot(df_rend, h)
    
    list_histo = pre_backtesting(liste_VaR_Tab[0], pf_returns, confidence_level)
    tree1.insert("", tk.END, values=("VaR historique", list_histo[0], list_histo[1], list_histo[2], list_histo[3]))
    
    list_histo = pre_backtesting(liste_VaR_Tab[1], pf_returns, confidence_level)
    tree1.insert("", tk.END, values=("VaR paramétrique", list_histo[0], list_histo[1], list_histo[2], list_histo[3]))
    
    list_histo = pre_backtesting(liste_VaR_Tab[2], pf_returns, confidence_level)
    tree1.insert("", tk.END, values=("VaR Garch", list_histo[0], list_histo[1], list_histo[2], list_histo[3]))
    
    list_histo = pre_backtesting(liste_VaR_Tab[3], pf_returns, confidence_level)
    tree1.insert("", tk.END, values=("VaR Cornish-Fischer", list_histo[0], list_histo[1], list_histo[2], list_histo[3]))
    
    list_histo = pre_backtesting(liste_VaR_Tab[4], pf_returns, confidence_level)
    tree1.insert("", tk.END, values=("VaR Riskmetrics", list_histo[0], list_histo[1], list_histo[2], list_histo[3]))
    
    list_histo = pre_backtesting(liste_VaR_Tab[5], pf_returns, confidence_level)
    tree1.insert("", tk.END, values=("VaR TVE", list_histo[0], list_histo[1], list_histo[2], list_histo[3]))
    
    list_histo = pre_backtesting(liste_VaR_Tab[6], pf_returns, confidence_level)
    tree1.insert("", tk.END, values=("VaR TVE-Garch", list_histo[0], list_histo[1], list_histo[2], list_histo[3]))
    
    tree1.place(x = 400, y = 250)

bouton_reporting = tk.Button(fenetre, text="Reporting", command=reporting)
bouton_reporting.place(x = 700, y = 662.5)

bouton_backtestig = tk.Button(fenetre, text="Backtesting", command=backtesting)
bouton_backtestig.place(x = 505, y = 662.5)

label_resultat = tk.Label(fenetre, text="")
label_resultat.pack()

fenetre.mainloop()

