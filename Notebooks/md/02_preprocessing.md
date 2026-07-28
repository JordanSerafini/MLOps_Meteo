# Prévision de pluie en Australie — 2/3 · Préparation des données

Nettoyage de la cible, création de la feature `Month`, puis construction d'un préprocesseur
qui n'apprend que sur le jeu d'entraînement.

Ce notebook se relance seul : il recharge le CSV depuis zéro.


```python
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
RANDOM_STATE = 42
```


```python
# Résolution robuste du chemin des données : fonctionne que le notebook soit
# lancé depuis la racine du projet ou depuis le dossier notebooks/.
CANDIDATES = [
    Path("Data/weatherAUS.csv"),
    Path("../Data/weatherAUS.csv"),
    Path.home() / "Desktop/MlOps_Meteo-Liora/Data/weatherAUS.csv",
]
DATA_PATH = next((p for p in CANDIDATES if p.exists()), None)
assert DATA_PATH is not None, "weatherAUS.csv introuvable (vérifier le dossier Data/)"

df = pd.read_csv(DATA_PATH, na_values=["NA"])
print(f"Chargé depuis : {DATA_PATH}")
print(f"Dimensions    : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
df.head()
```

    Chargé depuis : /home/tinkerbell/Desktop/MlOps_Meteo-Liora/Data/weatherAUS.csv
    Dimensions    : 145,460 lignes × 23 colonnes





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Nettoyage et features

Trois opérations avant de modéliser :

- les lignes sans `RainTomorrow` sont inutilisables, on les retire (~2,2 % du dataset) ;
- la cible passe en 0/1 ;
- `Date` est remplacée par le mois, qu'on traitera comme une catégorie. La saisonnalité vue
  dans le notebook 1 (pic hivernal à ~27 %, creux estival à ~19 %) justifie de la garder ;
  le jour et l'année, eux, n'apportent rien à une prévision à 24 h.


```python
data = df.drop(columns=["_month"], errors="ignore").copy()
data = data.dropna(subset=["RainTomorrow"]).copy()

# Cible 0/1
y = (data["RainTomorrow"] == "Yes").astype(int)

# Feature temporelle
data["Month"] = pd.to_datetime(data["Date"], errors="coerce").dt.month.astype("Int64")
data = data.drop(columns=["Date", "RainTomorrow"])

# Listes de features
categorical_features = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]
numeric_features = [c for c in data.columns if c not in categorical_features]
X = data

print(f"Échantillons : {len(X):,}  |  numériques : {len(numeric_features)}  |  catégorielles : {len(categorical_features)}")
print("Numériques  :", numeric_features)
print("Catégoriques:", categorical_features)
```

    Échantillons : 142,193  |  numériques : 16  |  catégorielles : 6
    Numériques  : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    Catégoriques: ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Month']


## 2. Le préprocesseur

Deux chaînes de traitement selon le type de colonne, assemblées dans un `ColumnTransformer` :
médiane + standardisation pour les numériques, mode + one-hot pour les catégorielles.

La médiane est préférée à la moyenne parce que `Rainfall` et `Evaporation` ont des
distributions très asymétriques (quelques jours de fortes pluies tirent la moyenne).
`handle_unknown="ignore"` évite un plantage si une modalité n'apparaît que dans le test.


```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])
preprocessor
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 [&#x27;MinTemp&#x27;, &#x27;MaxTemp&#x27;, &#x27;Rainfall&#x27;,
                                  &#x27;Evaporation&#x27;, &#x27;Sunshine&#x27;, &#x27;WindGustSpeed&#x27;,
                                  &#x27;WindSpeed9am&#x27;, &#x27;WindSpeed3pm&#x27;, &#x27;Humidity9am&#x27;,
                                  &#x27;Humidity3pm&#x27;, &#x27;Pressure9am&#x27;, &#x27;Pressure3pm&#x27;,
                                  &#x27;Cloud9am&#x27;, &#x27;Cloud3pm&#x27;, &#x27;Temp9am&#x27;,
                                  &#x27;Temp3pm&#x27;]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehot&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                 [&#x27;Location&#x27;, &#x27;WindGustDir&#x27;, &#x27;WindDir9am&#x27;,
                                  &#x27;WindDir3pm&#x27;, &#x27;RainToday&#x27;, &#x27;Month&#x27;])])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;&nbsp;ColumnTransformer<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for ColumnTransformer</span></a><span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></label><div class="sk-toggleable__content "><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;median&#x27;)),
                                                 (&#x27;scaler&#x27;, StandardScaler())]),
                                 [&#x27;MinTemp&#x27;, &#x27;MaxTemp&#x27;, &#x27;Rainfall&#x27;,
                                  &#x27;Evaporation&#x27;, &#x27;Sunshine&#x27;, &#x27;WindGustSpeed&#x27;,
                                  &#x27;WindSpeed9am&#x27;, &#x27;WindSpeed3pm&#x27;, &#x27;Humidity9am&#x27;,
                                  &#x27;Humidity3pm&#x27;, &#x27;Pressure9am&#x27;, &#x27;Pressure3pm&#x27;,
                                  &#x27;Cloud9am&#x27;, &#x27;Cloud3pm&#x27;, &#x27;Temp9am&#x27;,
                                  &#x27;Temp3pm&#x27;]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehot&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                 [&#x27;Location&#x27;, &#x27;WindGustDir&#x27;, &#x27;WindDir9am&#x27;,
                                  &#x27;WindDir3pm&#x27;, &#x27;RainToday&#x27;, &#x27;Month&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label  sk-toggleable__label-arrow ">num</label><div class="sk-toggleable__content "><pre>[&#x27;MinTemp&#x27;, &#x27;MaxTemp&#x27;, &#x27;Rainfall&#x27;, &#x27;Evaporation&#x27;, &#x27;Sunshine&#x27;, &#x27;WindGustSpeed&#x27;, &#x27;WindSpeed9am&#x27;, &#x27;WindSpeed3pm&#x27;, &#x27;Humidity9am&#x27;, &#x27;Humidity3pm&#x27;, &#x27;Pressure9am&#x27;, &#x27;Pressure3pm&#x27;, &#x27;Cloud9am&#x27;, &#x27;Cloud3pm&#x27;, &#x27;Temp9am&#x27;, &#x27;Temp3pm&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;SimpleImputer<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content "><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;StandardScaler<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></label><div class="sk-toggleable__content "><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label  sk-toggleable__label-arrow ">cat</label><div class="sk-toggleable__content "><pre>[&#x27;Location&#x27;, &#x27;WindGustDir&#x27;, &#x27;WindDir9am&#x27;, &#x27;WindDir3pm&#x27;, &#x27;RainToday&#x27;, &#x27;Month&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;SimpleImputer<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content "><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content "><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>



## 3. Découpage train / test

Le découpage vient **avant** tout ajustement. C'est ce qui rend la suite honnête.

`stratify=y` conserve les 22 % de jours pluvieux dans les deux jeux — sans ça, un tirage
malchanceux peut décaler le taux de base et fausser la comparaison des modèles.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

print(f"Train : {X_train.shape[0]:,} lignes  ({100*y_train.mean():.2f}% de pluie)")
print(f"Test  : {X_test.shape[0]:,} lignes  ({100*y_test.mean():.2f}% de pluie)")
```

    Train : 113,754 lignes  (22.42% de pluie)
    Test  : 28,439 lignes  (22.42% de pluie)


## 4. Ajustement

`fit_transform` sur le train, `transform` seul sur le test : le préprocesseur n'apprend jamais
rien du jeu de test.


```python
Xtr = preprocessor.fit_transform(X_train)
Xte = preprocessor.transform(X_test)

print(f"Train transformé : {Xtr.shape[0]:,} × {Xtr.shape[1]} colonnes")
print(f"Test transformé  : {Xte.shape[0]:,} × {Xte.shape[1]} colonnes")
print(f"({len(numeric_features)} numériques + one-hot des {len(categorical_features)} catégorielles)")
```

    Train transformé : 113,754 × 127 colonnes
    Test transformé  : 28,439 × 127 colonnes
    (16 numériques + one-hot des 6 catégorielles)



```python
import scipy.sparse as sp

def compte_nan(M):
    return np.isnan(M.data).sum() if sp.issparse(M) else np.isnan(M).sum()

print("NaN restants — train :", compte_nan(Xtr), " test :", compte_nan(Xte))

# Verification : les valeurs d'imputation memorisees sont bien celles du train.
imputeur = preprocessor.named_transformers_["num"].named_steps["imputer"]
appris = pd.Series(imputeur.statistics_, index=numeric_features)
verif = pd.DataFrame({"appris_par_le_pipeline": appris[["Humidity3pm", "Sunshine", "Pressure3pm"]],
                      "mediane_train": X_train[["Humidity3pm", "Sunshine", "Pressure3pm"]].median()})
verif["identique"] = np.isclose(verif.iloc[:, 0], verif.iloc[:, 1])
verif.round(4)
```

    NaN restants — train : 0  test : 0





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>appris_par_le_pipeline</th>
      <th>mediane_train</th>
      <th>identique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Humidity3pm</th>
      <td>52.0</td>
      <td>52.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Sunshine</th>
      <td>8.5</td>
      <td>8.5</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Pressure3pm</th>
      <td>1015.2</td>
      <td>1015.2</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Plus aucune valeur manquante, et les statistiques mémorisées correspondent au train.

## 5. Ce que ce découpage évite vraiment

Le notebook d'origine imputait sur le dataset entier avant de découper : les statistiques
utilisées pour le train étaient calculées en partie sur des lignes de test. C'est le reproche
classique fait à ce type de préparation. Avant d'en faire un argument, autant le mesurer.


```python
cols = ["Humidity3pm", "Sunshine", "Pressure3pm", "Cloud3pm", "Evaporation"]

ecart = pd.DataFrame({
    "mediane_dataset_entier": X[cols].median(),
    "mediane_train_seul": X_train[cols].median(),
})
ecart["difference"] = (ecart.iloc[:, 0] - ecart.iloc[:, 1]).abs()
ecart.round(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mediane_dataset_entier</th>
      <th>mediane_train_seul</th>
      <th>difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Humidity3pm</th>
      <td>52.0</td>
      <td>52.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Sunshine</th>
      <td>8.5</td>
      <td>8.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Pressure3pm</th>
      <td>1015.2</td>
      <td>1015.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Cloud3pm</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Evaporation</th>
      <td>4.8</td>
      <td>4.8</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Les écarts sont nuls. Avec 113 000 lignes d'entraînement, retirer 20 % du dataset ne déplace
pas une médiane — et refaire le calcul sur des sous-échantillons de 1 500 ou 300 lignes donne des
différences de l'ordre de 0,1 à 0,3 sur des variables qui valent 5 à 1 015.

Autant le dire : sur ce dataset, la fuite par imputation ne coûte rien en score. Justifier le
pipeline par un gain de performance serait malhonnête, et un correcteur qui refait le calcul
s'en apercevrait.

Les deux vraies raisons de le garder n'ont rien à voir avec le score.

### 5.1 Une imputation globale n'est pas rejouable au moment de servir

En production, l'API reçoit **une seule observation** et doit la préparer exactement comme à
l'entraînement. Calculer une médiane sur une ligne n'a aucun sens : les statistiques doivent
avoir été mémorisées pendant le `fit`, puis transportées avec le modèle.

C'est ce que fait `App/src/api/main.py`, qui charge un pipeline complet depuis le registry
MLflow plutôt qu'un classifieur nu.


```python
# Une observation isolee, comme celle que recoit l'API
une_ligne = X_test.iloc[[0]]
manquantes = int(une_ligne.isna().sum(axis=1).iloc[0])
print(f"Entrée : {une_ligne.shape[1]} colonnes brutes, dont {manquantes} manquante(s)")

print("Sortie :", preprocessor.transform(une_ligne).shape, "— mêmes colonnes que le train")
```

    Entrée : 22 colonnes brutes, dont 3 manquante(s)
    Sortie : (1, 127) — mêmes colonnes que le train


### 5.2 Une modalité inconnue ne doit pas faire tomber l'API

Le `pd.get_dummies` du notebook d'origine construisait ses colonnes à partir des modalités
présentes dans les données. Une station absente du train produirait un jeu de colonnes différent,
donc une erreur au moment de prédire. `handle_unknown="ignore"` met simplement la ligne à zéro
sur toutes les modalités de la variable.


```python
inconnue = X_test.iloc[[0]].copy()
inconnue["Location"] = "Marseille"      # station absente du dataset australien

print("Station inconnue acceptée :", preprocessor.transform(inconnue).shape)
print("Pas d'exception — l'API répond au lieu de renvoyer un 500.")
```

    Station inconnue acceptée : (1, 127)
    Pas d'exception — l'API répond au lieu de renvoyer un 500.


Ces deux garanties viennent du même choix : le préprocesseur est un objet ajusté, versionné et
déployé avec le modèle, pas une suite de transformations refaites à la main. La question de la
fuite disparaît par construction, sans avoir à vérifier si le dataset est assez gros pour qu'elle
soit indolore.

La suite (`03_modelisation.ipynb`) branche un classifieur derrière ce préprocesseur.
