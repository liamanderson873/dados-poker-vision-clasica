"""
tree.py - Arbol de decision EXPLICITO.

El arbol vive en config.json con este formato (editable a mano):
    nodo interno: {"feature": str, "threshold": float, "why": str,
                   "left": nodo  (feature <= threshold),
                   "right": nodo (feature >  threshold)}
    hoja:         {"label": str}

Cada nodo guarda su justificacion ("why"); calibrate.py anade las medidas
(margen entre clases) que respaldan el umbral.
"""


def classify(feats, node):
    """Recorre el arbol. Devuelve (clase, traza) donde la traza lista cada
    decision tomada: util para depurar y para explicar en la demo."""
    trace = []
    while "label" not in node:
        v = feats[node["feature"]]
        go_right = v > node["threshold"]
        trace.append(f"{node['feature']}={v:.3f} {'>' if go_right else '<='} {node['threshold']}")
        node = node["right"] if go_right else node["left"]
    return node["label"], trace


def to_text(node, indent=""):
    """Representacion en texto del arbol (para el informe)."""
    if "label" in node:
        return f"{indent}-> {node['label']}\n"
    s = f"{indent}[{node['feature']} <= {node['threshold']}]"
    if node.get("why"):
        s += f"   # {node['why']}"
    s += "\n"
    s += f"{indent}  si:\n" + to_text(node["left"], indent + "    ")
    s += f"{indent}  no:\n" + to_text(node["right"], indent + "    ")
    return s


def to_mermaid(node):
    """Diagrama Mermaid del arbol (pegar en el informe o en mermaid.live)."""
    lines = ["flowchart TD"]
    counter = [0]

    def walk(n):
        i = counter[0]
        counter[0] += 1
        if "label" in n:
            lines.append(f'    n{i}(["{n["label"]}"])')
            return i
        lines.append(f'    n{i}{{"{n["feature"]} &le; {n["threshold"]}?"}}')
        l = walk(n["left"])
        r = walk(n["right"])
        lines.append(f"    n{i} -- si --> n{l}")
        lines.append(f"    n{i} -- no --> n{r}")
        return i

    walk(node)
    return "\n".join(lines)


def features_used(node, acc=None):
    acc = acc if acc is not None else []
    if "label" not in node:
        if node["feature"] not in acc:
            acc.append(node["feature"])
        features_used(node["left"], acc)
        features_used(node["right"], acc)
    return acc
