"""
Fichier de fonctions utiles pour la problématique de l'APP6 (S2)
(c) JB Michaud, Sylvain Nicolay Université de Sherbrooke
v 1.0 Hiver 2023
v 1.1 - Corrigé un cas limite dans simplifytf
      - Utilisé des fonctions et une logique plus intuitive à lire dans simplifytf
      - Implémenté un workaround pour np.unwrap pour d'anciennes versions de numpy
      - Ajusté adéquatement l'utilisation de period= dans np.unwrap
      - Généralisé le code correctdelaybug au cas où, mais cette fonction ne devrait plus servir, a été mise en commentaire

Fonctions de visualisation
pzmap: affiche les pôles et les zéros déjà calculés
bode1: affiche un lieu de bode déjà calculé
bodeplot: calcule et affiche le lieu de bode d'une FT
grpdel1: affiche le délai de groupe déjà calculé
timeplt1: affiche une réponse temporelle déjà calculée
timepltmutlti1: affiche plusieurs réponses temporelles déjà calculées à différentes fréquences
timeplotmulti2: affiche plusieurs réponses temporelles déjà calculées pour différents systèmes

Fonctions de manipulation de FT
paratf: calcule la FT simpifiée équivalente à 2 FT en parallèle
seriestf: calcule la FT simplifiée équivalente à 2 FT en série (i.e. en cascade)
simplifytf: simplifie les pôles et les zéros d'une FT, et arrondis les parties réelles et imaginaires à l'entier lorsque pertinent
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


###############################################################################
def pzmap1(z, p, title):
    """
    Affiche les pôles et les zéros sur le plan complexe

    :param z: liste des zéros
    :param p: liste des pôles
    :param title: titre du graphique
    :return: handles des éléments graphiques générés
    """

    if len(p) == 0:     # safety check cas limite aucun pôle
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if len(z):
        ax.plot(np.real(z), np.imag(z), 'o', fillstyle='none', label='Zéros')  # affichage des zeros avec marqueurs 'o' ouverts
    ax.plot(np.real(p), np.imag(p), 'x', fillstyle='none', label='Pôles')  # affichage des poles avec des marqueurs 'x'
    fig.suptitle('Pôle/zéros de ' + title)
    ax.set_xlabel("Partie réelle ($Re(s)$)")
    ax.set_ylabel("Partie imaginaire ($Im(s)$)")
    # Recherche des min et max pour ajuster les axes x et y du graphique
    # longue histoire courte, concatène toutes les racines dans 1 seule liste, puis réserve une marge de chaque côté
    rootslist = []
    if len(z):
        rootslist.append(z)
    rootslist.append(p)
    rootslist = [item for sublist in rootslist for item in sublist]
    ax.set_xlim(np.amin(np.real(rootslist)) - .5, np.amax(np.real(rootslist)) + .5)
    ax.set_ylim(np.amin(np.imag(rootslist)) - .5, np.amax(np.imag(rootslist)) + .5)
    return fig, ax


###############################################################################
def bode1(w, mag, phlin, title):
    """
    Affiche le lieu un lieu de bode déjà calculé

    :param w: vecteur des fréquences du lieu de bode
    :param mag: vecteur des amplitudes, assumées en dB, doit être de même longueur que w
    :param phlin: vecteur des phases, assumées en degrés, doit être de même longueur que w
    :param title: titre du graphique
    :return: handles des éléments graphiques générés
    """

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    fig.suptitle(title + ' Frequency Response')

    ax[0].plot(w, mag)
    ax[0].set_xscale('log')
    ax[0].grid(visible=None, which='both', axis='both', linewidth=0.5)
    # fixe les limites du graphiques en gardant une marge minimale
    ax[0].set_xlim(10 ** (np.floor(np.log10(np.amin(w))) - 0.1), 10 ** (np.ceil(np.log10(np.amax(w))) + .1))
    ax[0].set_ylim(20 * (np.floor(np.amin(mag) / 20 - 0.1)), 20 * (np.ceil(np.amax(mag) / 20 + .1)))
    ax[0].set_ylabel('Amplitude [dB]')

    ax[1].plot(w, phlin)
    ax[1].set_xscale('log')
    ax[1].grid(visible=None, which='both', axis='both', linewidth=0.5)
    ax[1].set_xlabel('Frequency [rad/s]')
    ax[1].set_ylabel('Phase [deg]')
    # fixe les limites du graphiques en gardant une marge minimale
    ax[1].set_xlim(10 ** (np.floor(np.log10(np.amin(w))) - 0.1), 10 ** (np.ceil(np.log10(np.amax(w))) + .1))
    ax[1].set_ylim(20 * (np.floor(np.amin(phlin) / 20) - 1), 20 * (np.floor(np.amax(phlin) / 20) + 2))
    return fig, ax


###############################################################################
def bodeplot(b, a, title):
    """
    Calcule et affiche le lieu de bode d'une FT

    :param b: numérateur de la FT sous forme np.poly
    :param a: dénominateur de la FT sous forme np.poly
    :param title: titre du graphique
    :return: amplitude (dB) et phase (radians) calculés aux fréquences du vecteur w (rad/s) et les handles des éléments
        graphiques générés
    """

    w, h = signal.freqs(b, a, 5000)  # calcul la réponse en fréquence du filtre (H(jw)), fréquence donnée en rad/sec
    mag = 20 * np.log10(np.abs(h))
    ph = np.unwrap(np.angle(h), period=np.pi) if np.__version__ > '1.21' else \
        np.unwrap(2*np.angle(h))/2  # calcul du déphasage en radians
    phlin = np.rad2deg(ph)  # déphasage en degrés
    fig, ax = bode1(w, mag, phlin, title)
    return mag, ph, w, fig, ax


###############################################################################
def correctdelaybug(delay):
    """
    Corrige un glitch dans le calcul de la phase près des fréquences de coupure et des pôles
        lorsque la phase change de -pi à +pi (arctan2)
    Comme quoi python c'est pas matlab
    Vous pouvez ignorer cette fonction ça fait pas partie de la compétence de l'APP, si vous êtes curieux, allez
        voir le jump de phase (faut zoomer pas mal) près de w = 1 dans le bode de l'exemple de butterworth

    :param delay: vecteur des délais de groupe calculés
    :return: le délai de groupe sans le glitch
    """

    Done = False
    while not Done:
        index = np.argmin(delay)
        if 0 < index < len(delay) - 1:
            step = np.average([delay[index - 1], delay[index + 1]]) - delay[index]
        else:
            step = delay[index - 1] - delay[index] if index else delay[1] - delay[index]
        if step > .1:
            if 0 < index < len(delay) - 1:
                delay[index] = np.average([delay[index - 1], delay[index + 1]])
            else:
                delay[index] = delay[index - 1] if index else delay[1]
        else:
            Done = True
    Done = False
    while not Done:
        index = np.argmax(delay)
        if 0 < index < len(delay) - 1:
            step = np.average([delay[index - 1], delay[index + 1]]) - delay[index]
        else:
            step = delay[index - 1] - delay[index] if index else delay[1] - delay[index]
        if step < -.1:
            if 0 < index < len(delay) - 1:
                delay[index] = np.average([delay[index - 1], delay[index + 1]])
            else:
                delay[index] = delay[index - 1] if index else delay[1]
        else:
            Done = True
    return delay


###############################################################################
def grpdel1(w, delay, title):
    """
    Affiche le délai de groupe déjà calculé

    :param w: vecteur des fréquences, assumées en rad/s
    :param delay: vecteur des délais de groupe, assumé en secondes, doit être de longueur len(w)-1
    :param title: titre du graphique
    :return: handles des éléments graphiques générés
    """

    # delay = correctdelaybug(delay)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle('Group Delay ' + title)
    ax.plot(w[:len(w) - 1], delay)
    ax.set_xscale('log')
    ax.set_xlabel('Fréquence [rad/s]')
    ax.set_ylabel('Délai de groupe [s]')
    ax.grid(which='both', axis='both')
    ax.set_xlim(10 ** (np.floor(np.log10(np.amin(w))) - 0.1), 10 ** (np.ceil(np.log10(np.amax(w))) + .1))
    return fig, ax


###############################################################################
def timeplt1(t, u, tout, yout, title):
    """
    Affiche le résultat de  la simulation temporelle d'un système

    :param t: vecteur de temps en entrée de lsim, assumé en secondes
    :param u: vecteur d'entrée du système, doit être de même longueur que t
    :param tout: vecteur de temps en sortie de lsim, assumé en secondes
    :param yout: vecteur de réponse du système, doit être de même longueur que tout
    :return: handles des éléments graphiques générés
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle('Réponse temporelle '+title)
    ax.plot(t, u, 'r', alpha=0.5, linewidth=1, label='input')
    ax.plot(tout, yout, 'k', linewidth=1.5, label='output')
    ax.legend(loc='best', shadow=True, framealpha=1)
    ax.grid(alpha=0.3)
    ax.set_xlabel('t (s)')
    return fig, ax


###############################################################################
def timepltmulti1(t, u, w, tout, yout, title):
    """
    Affiche la réponse d'un même système à N entrées assumées sinusoîdales, chacune dans un subplot

    :param t: vecteur de temps fourni à lsim, assumé en secondes
    :param u: liste de N vecteurs d'entrée, doivent tous être de mpeme longueur que t
    :param w: liste de la fréquence des N sinusoîdes
    :param tout: vecteur de temps en sortie de lsim, assumé en secondes
    :param yout: liste de N vecteurs de sortie de lsim, doivent tous être de même longueur que tout
    :param title: titre du graphique
    :return: handles des éléments graphiques générés
    """

    fig, ax = plt.subplots(len(w), 1, figsize=(6, 6))
    fig.suptitle('Réponses temporelles de ' + title)
    for i in range(len(w)):
        ax[i].plot(t, u[i], 'r', alpha=0.5, linewidth=1, label=f'Input {w[i]} rad/s')
        ax[i].plot(tout[i], yout[i], 'k', linewidth=1.5, label=f'Output {w[i]} rad/s')
        ax[i].legend(loc='best', shadow=True, framealpha=1)
        ax[i].grid(alpha=0.3)
        if i == len(w) - 1:
            ax[i].set_xlabel('t (s)')
    return fig, ax


###############################################################################
def timepltmulti2(t, u, tout, yout, title, systems):
    """
    Affiche N résultats de simulation temporelle de N systèmes dans N subplots

    :param t: vecteur de temps fourni à lsim pour tous les systèmes, assumé en secondes
    :param u: vecteur d'entrée de tous les systèmes, doit être de même longueur que t
    :param tout: vecteur de temps en sortie de lsim pour tous les systèmes, assumé en secondes
    :param yout: liste de N vecteurs de sortie de lsim pour chacun des systèmes, chaque vecteur de même longueur que tout
    :param title: titre du graphique
    :param systems: liste de N noms des systèmes simulés
    :return: handles des éléments graphiques générés
    """

    fig, ax = plt.subplots(len(yout), 1, figsize=(6, 6))
    fig.suptitle('Réponses temporelles de ' + title)
    for i in range(len(yout)):
        ax[i].plot(t, u, 'r', alpha=0.5, linewidth=1, label=f'Input {systems[i]}')
        ax[i].plot(tout, yout[i], 'k', linewidth=1.5, label=f'Output {systems[i]}')
        ax[i].legend(loc='best', shadow=True, framealpha=1)
        ax[i].grid(alpha=0.3)
        if i == len(yout) - 1:
            ax[i].set_xlabel('t (s)')
    return fig, ax


###############################################################################
def paratf(z1, p1, k1, z2, p2, k2):
    """
    Calcule la FT résultante simplifiée des 2 FT fournies en argument en parallèle

    :param z1: zéros de la FT #1
    :param p1: pôles de la FT #1
    :param k1: gain de la FT #1, tel que retourné par signal.tf2zpk par exemple
    :param z2: idem FT #2
    :param p2:
    :param k2:
    :return: z, p, k simplifiés de la FT résultante
    """
    b1, a1 = signal.zpk2tf(z1, p1, k1)
    b2, a2 = signal.zpk2tf(z2, p2, k2)
    # en parallèle, il faut mettre sur dénominateur commun et faire le produit croisé au numérateur
    bleft = np.convolve(b1, a2) # calcule les 2 termes du numérateur
    bright = np.convolve(b2, a1)
    b = np.polyadd(bleft, bright)
    a = np.convolve(a1, a2)
    z, p, k = signal.tf2zpk(b, a)
    z, p, k = simplifytf(z, p, k)
    return z, p, k


###############################################################################
def seriestf(z1, p1, k1, z2, p2, k2):
    """
    Calcule la FT résultante simplifiée des 2 FT fournies en argument en cascade

    :param z1: zéros de la FT #1
    :param p1: pôles de la FT #1
    :param k1: gain de la FT #1, tel que retourné par signal.tf2zpk par exemple
    :param z2: idem FT #2
    :param p2:
    :param k2:
    :return: z, p, k simplifiés de la FT résultante
    """
    # Plus facile de travailler en polynôme?
    b1, a1 = signal.zpk2tf(z1, p1, k1)
    b2, a2 = signal.zpk2tf(z2, p2, k2)
    # en série les numérateurs et dénominateurs sont simplement multipliés
    b = np.convolve(b1, b2)  # convolve est équivalant à np.polymul()
    a = np.convolve(a1, a2)
    z, p, k = signal.tf2zpk(b, a)
    z, p, k = simplifytf(z, p, k)
    return z, p, k


###############################################################################
def simplifytf(z, p, k):
    """
    - simplifie les racines identiques entre les zéros et les pôles
    - arrondit les parties réelles et imaginaires de tous les termes à l'entier

    :param z: zéros de la FT à simplifier
    :param p: pôles de la FT à simplifier
    :param k: k de la FT à simplifier, tel que retournée par signal.tf2zpk par exemple
    :return: z, p, k simplifiés
    """

    tol = 1e-6  # tolérance utilisée pour déterminer si un pôle et un zéro sont identiques ou un nombre est entier

    # cast tout en complexe d'abord couvre les cas où z ou p est complètment réel pour les comparaisons qui suivent
    z = z.astype(complex)
    p = p.astype(complex)
    # algorithme de simplification des pôles et des zéros
    # compliqué pcq que la comparaison de nombres en points flottants ne se pythonify pas très bien
    # et que plusieurs cas limites (e.g. FT résultantes avec aucune racine) nécessitent des contorsions
    while len(p) and len(z):  # tant que le numérateur et le dénominateur contiennent encore des racines
        match = False
        for i, zval in enumerate(z[:]):     # itère sur les zéros
            for j, pval in enumerate(p[:]):     # itère sur les pôles
                if np.isclose(zval, pval, atol=tol, rtol=tol):  # si le zéro est identique au pôle
                    p = np.delete(p, j)     # enlève ce zéro et ce pôle
                    z = np.delete(z, i)
                    match = True    # poutine pour repartir la recherche en cas de match, (pour les cas limites)
                    break
            if match:
                break
        else:
            break
    # itère sur les zéros, les pôles et enfin le gain pour arrondir à l'unité lorsque pertinent
    for i, zval in enumerate(z):
        if np.isclose(zval.real, np.round(zval.real), atol=tol, rtol=tol):   # teste si la valeur est identique à un entier
            z[i] = complex(np.round(z[i].real), z[i].imag)
        if np.isclose(zval.imag, np.round(zval.imag), atol=tol, rtol=tol):
            z[i] = complex(z[i].real, np.round(z[i].imag))
    for i, pval in enumerate(p):
        if np.isclose(pval.real, np.round(pval.real), atol=tol, rtol=tol):   # teste si la valeur est identique à un entier
            p[i] = complex(np.round(p[i].real), p[i].imag)
        if np.isclose(pval.imag, np.round(pval.imag), atol=tol, rtol=tol):
            p[i] = complex(p[i].real, np.round(p[i].imag))
    if np.isclose(k, np.round(k), atol=tol, rtol=tol):
        k = np.round(k)
    return z, p, k
