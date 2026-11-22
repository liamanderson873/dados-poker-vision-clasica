"""
hand.py - Nivel 3: interpretacion de la escena (jugada de poker con dados).

Jerarquia (de mayor a menor):
    Repoker (5 iguales) > Poker (4) > Full (3+2) > Escalera (5 consecutivas)
    > Trio > Doble pareja > Pareja > Nada

Escaleras posibles con las caras de nuestros dados (7-8-J-Q-K-A):
    menor: 7 8 J Q K    mayor: 8 J Q K A
(el orden de valor se toma de 'classes' en config.json)
Si en el tapete no hay exactamente 5 dados se informa igualmente de la mejor
combinacion presente, marcando la escena como incompleta.
"""
from collections import Counter

RANK = {"Repoker": 8, "Poker": 7, "Full": 6, "Escalera": 5, "Trio": 4,
        "Doble pareja": 3, "Pareja": 2, "Nada": 1, "Sin dados": 0}


def evaluate_hand(labels, classes):
    """labels: lista de clases detectadas. classes: orden de valor (menor a mayor).
    Devuelve dict(name, rank, detail, complete, counts)."""
    labels = [l for l in labels if l in classes]
    n = len(labels)
    if n == 0:
        return {"name": "Sin dados", "rank": 0, "detail": "", "complete": False, "counts": {}}
    cnt = Counter(labels)
    # Grupos ordenados por tamano y despues por valor de la cara
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], classes.index(kv[0])), reverse=True)
    sizes = [g[1] for g in groups]
    idx = sorted(classes.index(l) for l in labels)

    if sizes[0] >= 5:
        name = "Repoker"
    elif sizes[0] == 4:
        name = "Poker"
    elif sizes[0] == 3 and len(sizes) > 1 and sizes[1] >= 2:
        name = "Full"
    elif n == 5 and len(cnt) == 5 and idx[-1] - idx[0] == 4:
        name = "Escalera"
    elif sizes[0] == 3:
        name = "Trio"
    elif sizes[0] == 2 and len(sizes) > 1 and sizes[1] == 2:
        name = "Doble pareja"
    elif sizes[0] == 2:
        name = "Pareja"
    else:
        name = "Nada"

    if name == "Escalera":
        tramo = f"{classes[idx[0]]}-{classes[idx[-1]]}"
        detail = f"mayor ({tramo})" if idx[0] == 1 else f"menor ({tramo})"
    elif name == "Nada":
        detail = "carta alta " + classes[idx[-1]]
    else:
        detail = " + ".join(f"{c}x{k}" for k, c in groups if c >= 2)
    return {"name": name, "rank": RANK[name], "detail": detail,
            "complete": n == 5, "counts": dict(cnt)}
