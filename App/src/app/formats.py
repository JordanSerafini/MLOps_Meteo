"""Formatage des nombres à la française.

Le support est présenté et noté en français : on écrit 0,32 et 28 439, pas 0.32 et 28,439.
Ces fonctions sont centralisées parce qu'elles servent dans les quatre onglets, et qu'un
mélange des deux conventions sur le même écran se remarque immédiatement.
"""


def nb(valeur, decimales=2):
    """0.3245 -> « 0,32 »."""
    return f"{valeur:.{decimales}f}".replace(".", ",")


def ent(valeur):
    """28439 -> « 28 439 » (espace insécable fine, pas de virgule des milliers)."""
    return f"{valeur:,}".replace(",", " ")


def pct(valeur, decimales=1, sur_cent=False):
    """0.2242 -> « 22,4 % ». sur_cent=True si la valeur est déjà en pourcentage."""
    valeur = valeur if sur_cent else valeur * 100
    return f"{nb(valeur, decimales)} %"


def signe(valeur, decimales=3):
    """Delta d'une métrique : « +0,173 » ou « -0,121 ».

    Le tiret reste le trait d'union ASCII, pas le signe moins typographique : `st.metric`
    lit le premier caractère pour choisir la flèche et la couleur, et ne reconnaît que « - ».
    """
    return f"{valeur:+.{decimales}f}".replace(".", ",")
