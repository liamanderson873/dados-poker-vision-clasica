"""
vision.py - Nucleo del sistema de vision artificial clasica para dados de poker.

Pipeline:
    Imagen RGB -> preprocesamiento -> segmentacion (tapete verde en Lab)
    -> deteccion de dados (contornos + filtros geometricos + separacion de dados en contacto)
    -> rectificacion de la cara superior -> segmentacion del simbolo
    -> extraccion de caracteristicas explicitas -> arbol de decision (tree.py)
    -> identificacion -> interpretacion de la jugada (hand.py)

No se usa ningun modelo aprendido: solo operaciones de imagen y reglas.
"""
import json
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    """Carga config.json (parametros y arbol de decision)."""
    with open(path or os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Preprocesamiento
# ---------------------------------------------------------------------------
def normalize_exposure(img, cfg):
    """Compensa subexposicion midiendo plastico claro, poco saturado y no verde.

    No usa el brillo del fieltro ni el carton como referencia. La ganancia es
    comun a los tres canales para preservar las proporciones de color.
    """
    p = cfg.get("exposure", {})
    if not p.get("enabled", False):
        return img, 1.0
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * min(1, 800/w))),
                             max(1, int(h * min(1, 800/w)))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    white = ((hsv[:,:,1] <= 60) & (hsv[:,:,2] >= 90) &
             (lab[:,:,1].astype(np.int16) - 128 >= -2))
    if np.count_nonzero(white) < 20:
        return img, 1.0
    reference = float(np.percentile(hsv[:,:,2][white], 90))
    if reference >= p.get("low_reference", 210):
        return img, 1.0
    gain = min(p.get("max_gain", 2.0), p.get("target", 240) / max(reference, 1))
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8), gain


def preprocess(img, cfg):
    """Reduce la imagen a un ancho maximo (velocidad) y aplica un suavizado
    gaussiano para atenuar ruido del sensor y la textura del fieltro."""
    p = cfg["preproc"]
    h, w = img.shape[:2]
    scale = 1.0
    if w > p["max_width"]:
        scale = p["max_width"] / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    k = p["blur_ksize"] | 1  # el kernel debe ser impar
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    return img, blurred, scale


# ---------------------------------------------------------------------------
# 2. Segmentacion: todo lo que NO es tapete verde es candidato a objeto
# ---------------------------------------------------------------------------
def felt_mask(blurred, cfg):
    """Mascara binaria del tapete en espacio Lab (OpenCV: L 0-255, a* centrado en 0).
    Medido en las fotos reales (calibrate.py felt):
      - fieltro: a* entre -15 y -6 tanto en zona iluminada como en sombra, L <= ~185
      - dado blanco: a* >= +8 (plastico blanco bajo luz calida), incluso en sombra
      - tinta roja: a* >> 0
    Un pixel es OBJETO si es claro y no verde (a* > a_felt_max y L >= l_obj_low)
    o si es muy luminoso (L >= l_obj_min; cubre dados bajo luz fria, a* ~ 0).
    El limite l_obj_low elimina la sombra que proyectan los dados: es fieltro
    oscuro (L ~ 50-90) cuyo a* sube hasta ~0 y si no se colaba como objeto.
    En HSV no funcionaba: el fieltro real tiene saturacion baja (S ~ 35-90)."""
    f = cfg["felt_lab"]
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.int16) - 128
    # Segunda regla: muy luminoso Y no claramente verde. Sin la condicion de a*,
    # el centro del fieltro iluminado por el flash (L ~ 205, a* ~ -15) pasaba por objeto.
    obj = ((a > f["a_felt_max"]) & (L >= f["l_obj_low"])) | ((L >= f["l_obj_min"]) & (a > f["a_bright_min"]))
    return np.uint8(~obj) * 255


def felt_roi(felt, kernel, margin_iterations=2):
    """Zona de trabajo: solo cuenta lo que esta DENTRO del tapete (envolvente
    convexa de la mayor mancha de fieltro). Asi se ignora la mesa, la sabana
    o la mano que aparezcan fuera del tapete en los bordes de la foto."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(felt, 8)
    if n <= 1:
        return np.full_like(felt, 255)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours(np.uint8(lbl == big) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(felt)
    cv2.drawContours(roi, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, cv2.FILLED)
    return cv2.erode(roi, kernel, iterations=margin_iterations) if margin_iterations else roi


def _fill_holes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled


def object_mask(blurred, cfg):
    """Invierte la mascara del tapete y limpia con morfologia:
    apertura (elimina motas de pelusa y separa dados unidos por 1-2 px).
    NO se usa cierre: uniria dados proximos rellenando el hueco de tapete
    entre ellos. Los huecos interiores (simbolo, reflejos) se rellenan.

    Dados pegados: la junta entre dos dados es una linea oscura y fina dentro
    de la zona clara. El 'black-hat' (cierre - imagen) sobre L la resalta; se
    elimina de la mascara y asi la mancha se parte por las juntas. Los trazos
    del simbolo tambien salen en el black-hat, pero quedan dentro del dado y
    se recuperan al rellenar huecos."""
    s = cfg["segmentation"]
    k = s["morph_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    felt = felt_mask(blurred, cfg)
    roi = felt_roi(felt, kernel, s.get("roi_margin_iterations", 2))
    raw = cv2.bitwise_not(felt)
    obj = cv2.bitwise_and(raw, roi)
    if s.get("retain_edge_components", False):
        # El dado puede ocultar una esquina del tapete: la envolvente del verde
        # corta entonces el dado en diagonal. Conservar su componente completa
        # si una parte suficiente esta anclada al tapete. El carton exterior,
        # sin ese solapamiento, no se incorpora.
        clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            patch = np.zeros((h, w), np.uint8)
            cv2.drawContours(patch, [contour - [x, y]], -1, 255, cv2.FILLED)
            overlap = cv2.countNonZero(cv2.bitwise_and(patch, roi[y:y+h, x:x+w]))
            if overlap >= 0.3 * cv2.countNonZero(patch):
                cv2.drawContours(obj, [contour], -1, 255, cv2.FILLED)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel)
    obj = _fill_holes(obj)

    side = _estimate_side(obj)
    if side > 0 and s.get("seam_blackhat_min", 0) > 0:
        L = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)[:, :, 0]
        kb = max(3, int(side * s["seam_kernel_frac"])) | 1
        bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))
        seams = np.uint8(bh >= s["seam_blackhat_min"]) * 255
        cut = cv2.bitwise_and(obj, cv2.bitwise_not(seams))
        cut = cv2.morphologyEx(cut, cv2.MORPH_OPEN, kernel)
        obj = _fill_holes(cut)
    return obj


def _estimate_side(mask):
    """Lado de UN dado en px aunque esten pegados: 2 * maximo de la transformada
    de distancia de cada mancha (vale lado/2 para un dado suelto y tambien para
    una fila de dados). Se toma la mediana de las manchas grandes."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sides = []
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        blob = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(blob, [c - [x - 1, y - 1]], -1, 255, cv2.FILLED)
        sides.append(2 * cv2.distanceTransform(blob, cv2.DIST_L2, 5).max())
    if not sides:
        return 0.0
    # Ignorar tiras finas de ruido (bordes, pelusa): mucho mas estrechas que un dado
    big = [s for s in sides if s >= 0.5 * max(sides)]
    return float(np.median(big))


# ---------------------------------------------------------------------------
# 3. Deteccion de dados
# ---------------------------------------------------------------------------
def _split_touching(blob_mask, n_expected):
    """Separa dados en contacto con transformada de distancia + watershed.
    Los centros de los dados son los maximos de la distancia al borde."""
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    # Se prueban varios niveles de corte y se elige el que da un numero de
    # nucleos mas cercano al numero de dados esperado (area blob / area dado).
    best = None
    for frac in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
        _, pk = cv2.threshold(dist, frac * dist.max(), 255, cv2.THRESH_BINARY)
        pk = pk.astype(np.uint8)
        nn, mk = cv2.connectedComponents(pk)
        if best is None or abs(nn - 1 - n_expected) < abs(best[0] - 1 - n_expected):
            best = (nn, mk, pk)
    n, markers, peaks = best
    if n - 1 < 2:
        return [blob_mask]
    markers = markers + 1
    markers[(blob_mask > 0) & (peaks == 0)] = 0
    color = cv2.cvtColor(blob_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers.astype(np.int32))
    parts = []
    for lbl in range(2, n + 1):
        m = np.uint8(markers == lbl) * 255
        if cv2.countNonZero(m):
            parts.append(m)
    return parts


def _split_row(contour, side, n):
    """Regla para una FILA de n dados pegados (caso tipico al alinearlos):
    la mancha es un rectangulo de lado corto ~ 1 dado y lado largo ~ n dados.
    Se divide el rectangulo minimo en n trozos iguales a lo largo del eje largo."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour)
    if w < h:  # que 'w' sea siempre el lado largo
        w, h, ang = h, w, ang + 90
    ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    parts = []
    for i in range(n):
        off = (i + 0.5) * w / n - w / 2
        c = (cx + off * ux, cy + off * uy)
        box = cv2.boxPoints((c, (w / n, h), ang)).astype(np.int32)
        parts.append(box.reshape(-1, 1, 2))
    return parts


def _contour_props(c):
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    aspect = max(w, h) / max(1e-6, min(w, h))
    hull = cv2.convexHull(c)
    solidity = area / max(1e-6, cv2.contourArea(hull))
    return area, rect, aspect, solidity


def detect_dice(blurred, cfg, debug=None):
    """Devuelve lista de dados: dict(contour, rect, area, center).
    Filtros (justificar en calibrate.py):
      - area relativa a la de un dado (mediana o valor fijo)
      - relacion de aspecto del rectangulo minimo (un dado visto desde arriba ~1)
      - solidez (area / area envolvente convexa): un cuadrado ~1, objetos raros < 0.8
    """
    s = cfg["segmentation"]
    mask = object_mask(blurred, cfg)
    if debug is not None:
        debug["object_mask"] = mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 100]
    # Un dado valido sale entero en la foto: lo que toca el borde de la imagen
    # (pared o suelo de la caja que asoma, dado cortado) queda fuera de la zona util.
    H, W = mask.shape
    m = s.get("border_margin_px", 3)

    # Tampoco cuenta un dado encima del borde del tapete: la zona de trabajo se
    # recorta por el tapete, asi que ese dado queda cortado y solo se veria un
    # trozo (a menudo un lateral). Se detecta porque toca el limite de la zona.
    k = s["morph_kernel"]
    roi = felt_roi(felt_mask(blurred, cfg), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                   s.get("roi_margin_iterations", 2))
    outside = cv2.dilate(cv2.bitwise_not(roi), np.ones((5, 5), np.uint8))
    edge_max = s.get("felt_edge_max_frac", 0.0)

    def inside(c):
        x, y, w, h = cv2.boundingRect(c)
        if not s.get("allow_image_edge", False) and not (x > m and y > m and x + w < W - m and y + h < H - m):
            return False
        # Fraccion del contorno del dado que va pegada al limite del tapete
        pts = c.reshape(-1, 2)
        touching = outside[pts[:, 1], pts[:, 0]] > 0
        return touching.mean() <= edge_max

    cnts = [c for c in cnts if inside(c)]
    if not cnts:
        return []
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cnts, -1, 255, cv2.FILLED)

    ref = cfg["die"].get("expected_area_px") or 0
    if ref <= 0:
        # Area de UN dado aunque esten todos pegados (ver _estimate_side)
        ref = _estimate_side(mask) ** 2 * 0.97  # 0.97: esquinas redondeadas

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > s["touching_area_frac"] * ref:
            # Blob demasiado grande: probablemente varios dados en contacto
            n_exp = max(2, int(round(area / ref)))
            side = math.sqrt(ref)
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            long_, short = max(w, h), min(w, h)
            n_row = int(round(long_ / side))
            if abs(short / side - 1) < 0.35 and n_row >= 2 and abs(long_ / side - n_row) < 0.35:
                candidates.extend(_split_row(c, side, n_row))  # fila de dados
                continue
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
            for part in _split_touching(blob, n_exp):
                pc, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if pc:
                    candidates.append(max(pc, key=cv2.contourArea))
        else:
            candidates.append(c)

    dice = []
    for c in candidates:
        area, rect, aspect, solidity = _contour_props(c)
        if s.get("hull_rescue_min", 0) <= solidity < s["solidity_min"]:
            # Mancha "mordida": el corte por juntas se ha comido trazos del simbolo
            # que llegaban al borde (dado pequeno o visto en perspectiva). Se prueba
            # con su envolvente convexa; luego tiene que pasar igualmente los filtros
            # de tamano, forma, cara clara y simbolo.
            c = cv2.convexHull(c)
            area, rect, aspect, solidity = _contour_props(c)
        ok = (s["die_area_min_frac"] * ref <= area <= s["die_area_max_frac"] * ref
              and aspect <= s["aspect_max"] and solidity >= s["solidity_min"])
        if ok:
            M = cv2.moments(c)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            dice.append({"contour": c, "rect": rect, "area": area,
                         "aspect": aspect, "solidity": solidity, "center": (cx, cy)})
    # Orden de lectura: arriba-abajo, izquierda-derecha (para numerar en pantalla)
    dice.sort(key=lambda d: (round(d["center"][1] / math.sqrt(ref)), d["center"][0]))
    return dice


# ---------------------------------------------------------------------------
# 4. Rectificacion de la cara superior
# ---------------------------------------------------------------------------
def _axes(rect):
    """Ejes unitarios del rectangulo minimo: u va con el ancho, v con el alto."""
    a = math.radians(rect[2])
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def top_face_rects(contours, shape, cfg):
    """Cara de arriba de cada dado, quitando los laterales vistos en perspectiva.

    Modelo: la camara mira desde encima del centro de la foto. Lejos del centro
    se ve, ademas de la cara de arriba, el lateral que mira hacia el centro. En
    cada eje del rectangulo minimo del dado, su extension es
        E = s + k * |r|
    con s = lado de la cara de arriba, r = distancia del dado al centro de la
    foto medida sobre ese eje y k la proporcion de lateral que se ve (depende de
    la altura de la camara). s y k se ajustan por minimos cuadrados con TODOS los
    dados de la foto, asi que no hace falta conocer la altura. Luego, en cada
    dado se toma el rectangulo s x s del extremo mas lejano al centro."""
    p = cfg.get("perspective", {})
    rects = [cv2.minAreaRect(c) for c in contours]
    if not p.get("enabled", False) or len(rects) < p.get("min_dice", 3):
        return rects
    rows, cols = shape
    o = np.array([cols / 2, rows / 2])
    # Lado de la cara de arriba: el lado menor tipico de los dados de la foto
    # (un dado desplazado en una sola direccion solo muestra lateral en esa
    # direccion, asi que su lado corto es la cara de arriba).
    s = float(np.percentile([min(w, h) for (_, (w, h), _) in rects], p.get("side_percentile", 25)))
    out = []
    for (c, (w, h), ang) in rects:
        u, v = _axes((c, (w, h), ang))
        r = np.array(c) - o
        # Lado de la cara de arriba en cada eje: lo que sobra es lateral. Si lo
        # que sobra es poco (dado casi cenital), no se recorta en ese eje.
        local_side = min(s, w, h) if p.get("local_side", False) else s
        mn = p.get("min_lateral_frac", 0.0) * local_side
        minimum = 0.5 if p.get("local_side", False) else 0.6
        su = float(np.clip(local_side, minimum * w, w)) if w - local_side > mn else w
        sv = float(np.clip(local_side, minimum * h, h)) if h - local_side > mn else h
        # El lateral esta en el lado que mira al centro: la cara de arriba se
        # desplaza hacia fuera la mitad de lo que se recorta
        cc = (np.array(c) + np.sign(r @ u) * (w - su) / 2 * u + np.sign(r @ v) * (h - sv) / 2 * v)
        out.append(((float(cc[0]), float(cc[1])), (su, sv), ang))
    return out

def rectify_face(img, die, cfg, with_support=False):
    """Proyecta el rectangulo minimo del dado a un cuadrado de size_px x size_px
    (normaliza escala y rotacion) y recorta el centro (inner_frac) para descartar
    las caras laterales visibles y los bordes redondeados."""
    f = cfg["face"]
    S = f["size_px"]
    # boxPoints devuelve las 4 esquinas en orden ciclico. Se mapean tal cual a
    # un cuadrado: la cara queda girada un multiplo de 90 grados, lo cual no
    # importa porque todas las caracteristicas son invariantes a la rotacion.
    # (Ordenar por suma/diferencia de coordenadas falla con dados a ~45 grados.)
    box = cv2.boxPoints(die["rect"]).astype(np.float32)
    dst = np.array([[0, S - 1], [0, 0], [S - 1, 0], [S - 1, S - 1]], np.float32)
    H = cv2.getPerspectiveTransform(box, dst)
    # Lo que queda fuera del contorno de ESTE dado (tapete o dado vecino en
    # contacto) se pinta de blanco para que no se confunda con el simbolo.
    # Se usa la envolvente convexa: el contorno puede tener muescas donde el
    # corte por juntas se comio trazos del simbolo, y no hay que borrarlos.
    own = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(own, [cv2.convexHull(die["contour"])], -1, 255, cv2.FILLED)
    # ...pero sin meterse en los dados vecinos (dados pegados en fila)
    if die.get("others"):
        cv2.drawContours(own, die["others"], -1, 0, cv2.FILLED)
    src = img.copy()
    src[own == 0] = 255
    face = cv2.warpPerspective(src, H, (S, S), flags=cv2.INTER_LINEAR,
                               borderValue=(255, 255, 255))
    m = int(S * (1 - f["inner_frac"]) / 2)
    face = face[m:S - m, m:S - m]
    if with_support:
        support = cv2.warpPerspective(own, H, (S, S), flags=cv2.INTER_NEAREST,
                                      borderValue=0)
        return face, support[m:S - m, m:S - m] > 0
    return face


def face_white_score(img, contour, cfg):
    """Calidad fisica de un recorte alternativo: plastico blanco real.

    Mide el original sin pintar de blanco fuera del contorno. Sirve para
    elegir entre dos segmentaciones del mismo objeto, nunca entre clases.
    """
    local = dict(cfg)
    local["perspective"] = dict(cfg["perspective"], min_dice=1, local_side=True)
    rect = top_face_rects([contour], img.shape[:2], local)[0]
    box = cv2.boxPoints(rect).astype(np.float32)
    dst = np.array([[0,127],[0,0],[127,0],[127,127]], np.float32)
    transform = cv2.getPerspectiveTransform(box, dst)
    face = cv2.warpPerspective(img, transform, (128,128))[12:116,12:116]
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    return float(np.mean((lab[:,:,0] >= cfg["face"]["face_l_min"]) &
                         (hsv[:,:,1] <= cfg["face"].get("background_s_max",80)) &
                         (lab[:,:,1].astype(np.int16)-128 >= cfg["face"].get("background_a_min",-5))))


# ---------------------------------------------------------------------------
# 5. Segmentacion del simbolo dentro de la cara
# ---------------------------------------------------------------------------
def symbol_mask(face, cfg):
    """El cuerpo del dado es claro y poco saturado. La 'tinta' de un pixel se
    define como max(S, 255 - V): alta si es saturado (color) o oscuro (negro).
    Umbral = max(Otsu, ink_min): Otsu se adapta a la iluminacion, el minimo
    evita segmentar ruido cuando la cara esta casi vacia."""
    f = cfg["face"]
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    ink = np.maximum(hsv[:, :, 1], 255 - hsv[:, :, 2])
    ink = cv2.GaussianBlur(ink, (3, 3), 0)
    otsu, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu, f["ink_min"])
    _, m = cv2.threshold(ink, thr, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Reparar cortes de 1-2 px en trazos causados por reflejos/umbralizacion.
    close = f.get("symbol_close_px", 0)
    if close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    # Eliminar componentes diminutas (polvo, ruido)
    min_px = f["min_component_frac"] * m.size
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    clean = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            clean[lbl == i] = 255
    return clean, hsv, thr


# ---------------------------------------------------------------------------
# 6. Extraccion de caracteristicas
# ---------------------------------------------------------------------------
def _in_ranges(h, ranges):
    out = np.zeros(h.shape, bool)
    for lo, hi in ranges:
        out |= (h >= lo) & (h <= hi)
    return out


FEATURE_DOC = {
    "ink_ratio": "Area del simbolo / area de la cara rectificada",
    "n_comp": "Numero de componentes conexas del simbolo (8-conectividad)",
    "n_main_comp": "Componentes con area de pixeles >=20% de la componente mayor",
    "n_holes": "Numero de agujeros (contornos hijos en RETR_CCOMP) con area minima",
    "hole_ratio": "Area total de agujeros / area del simbolo",
    "euler": "Numero de Euler = n_comp - n_holes",
    "circularity": "4*pi*A/P^2 de la componente mayor",
    "aspect": "Lado largo / lado corto del rectangulo minimo de la componente mayor",
    "solidity": "Area / area de la envolvente convexa de la componente mayor",
    "solidity_closed": "Solidez tras cierre geometrico configurable: elipse 15x15 en config.json",
    "extent": "Area / area del rectangulo minimo de la componente mayor",
    "eccentricity": "Excentricidad de la elipse equivalente (momentos centrales)",
    "hu1": "-sign*log10|Hu1| del simbolo completo (invariante a rotacion/escala)",
    "hu2": "-sign*log10|Hu2| del simbolo completo",
    "red_frac": "Fraccion de pixeles del simbolo rojos (Lab: a* >= red_a_min)",
    "blue_frac": "Fraccion de pixeles del simbolo azules (Lab: b* <= blue_b_max)",
    "dark_frac": "Fraccion de pixeles del simbolo ni rojos ni azules (negro/gris)",
    "mean_a": "a* medio del simbolo (rojo >> 0, negro ~ 0)",
    "n_colors": "Numero de familias de color con fraccion > 0.15",
}


def select_top_ink(face, mask, outward, cfg):
    """Separa tintas de dos caras por componentes, color y direccion radial.

    Solo actua cuando hay grupos rojo/negro sustanciales, separados en la
    direccion hacia fuera de la imagen. Conserva componentes completas del
    grupo exterior, sin usar etiquetas ni plantillas de simbolos.
    """
    if outward is None:
        return mask, False
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    if n < 3:
        return mask, False
    red = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)[:,:,1].astype(np.int16)-128 >= cfg['color']['red_a_min']
    groups = {False: [], True: []}
    for k in range(1,n):
        groups[bool(np.mean(red[labels==k])>=.5)].append(k)
    if not all(groups.values()):
        return mask, False
    areas = {key:sum(stats[k,cv2.CC_STAT_AREA] for k in group) for key,group in groups.items()}
    if min(areas.values()) < .15 * sum(areas.values()):
        return mask, False
    centroids = {key:sum(centers[k]*stats[k,cv2.CC_STAT_AREA] for k in group)/areas[key] for key,group in groups.items()}
    delta = float((centroids[True]-centroids[False])@outward)
    if abs(delta)<.2*min(mask.shape):
        return mask, False
    keep = groups[delta>0]
    return np.uint8(np.isin(labels,keep))*255, True


def extract_features(face, cfg, outward=None):
    """Calcula el vector de caracteristicas explicitas de una cara rectificada.
    Todas son invariantes a la rotacion del dado (requisito de la prueba)."""
    f, col = cfg["face"], cfg["color"]
    m, hsv, thr = symbol_mask(face, cfg)
    m, top_filtered = select_top_ink(face, m, outward, cfg)
    total = m.size
    area = cv2.countNonZero(m)
    feats = {k: 0.0 for k in FEATURE_DOC}
    feats['top_ink_filtered'] = float(top_filtered)
    feats["ink_thr"] = float(thr)
    if area == 0:
        return feats, m
    feats["ink_ratio"] = area / total
    _, _, component_stats, _ = cv2.connectedComponentsWithStats(m, 8)
    sizes = component_stats[1:, cv2.CC_STAT_AREA]
    feats["n_main_comp"] = float(np.count_nonzero(sizes >= f.get("relative_component_min", .2) * sizes.max()))

    # Componentes y agujeros con jerarquia de contornos (2 niveles)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hier = hier[0]
    outer = [i for i in range(len(cnts)) if hier[i][3] == -1]
    min_hole = f["min_hole_frac"] * total
    holes = [i for i in range(len(cnts)) if hier[i][3] != -1 and cv2.contourArea(cnts[i]) >= min_hole]
    feats["n_comp"] = float(len(outer))
    feats["n_holes"] = float(len(holes))
    feats["hole_ratio"] = sum(cv2.contourArea(cnts[i]) for i in holes) / area
    feats["euler"] = feats["n_comp"] - feats["n_holes"]

    # Geometria de la componente mayor
    big = max((cnts[i] for i in outer), key=cv2.contourArea)
    A = cv2.contourArea(big)
    P = cv2.arcLength(big, True)
    feats["circularity"] = 4 * math.pi * A / (P * P) if P > 0 else 0.0
    (w, h) = cv2.minAreaRect(big)[1]
    feats["aspect"] = max(w, h) / max(1e-6, min(w, h))
    feats["extent"] = A / max(1e-6, w * h)
    feats["solidity"] = A / max(1e-6, cv2.contourArea(cv2.convexHull(big)))
    # Solo es una caracteristica geometrica: no se cambia el conteo de puntos
    # ni el color. Evita que una Q con un pequeno corte parezca una J abierta.
    repair = f.get("geometry_close_px", 5)
    kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (repair, repair))
              if f.get("geometry_close_shape", "rect") == "ellipse"
              else np.ones((repair, repair), np.uint8))
    repaired = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    repaired_contours, _ = cv2.findContours(repaired, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    repaired_big = max(repaired_contours, key=cv2.contourArea)
    feats["solidity_closed"] = cv2.contourArea(repaired_big) / max(1e-6, cv2.contourArea(cv2.convexHull(repaired_big)))

    # Momentos del simbolo completo
    M = cv2.moments(m, binaryImage=True)
    mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
    common = math.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    feats["eccentricity"] = math.sqrt(max(0.0, 1 - l2 / l1)) if l1 > 0 else 0.0
    hu = cv2.HuMoments(M).ravel()
    feats["hu1"] = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-30))
    feats["hu2"] = float(-np.sign(hu[1]) * np.log10(abs(hu[1]) + 1e-30))

    # Color de los pixeles del simbolo en Lab. Con HSV la tinta negra salia
    # "roja": en pixeles oscuros el tono H es inestable y la luz calida lo
    # empuja hacia el rojo. En Lab el rojo es a* muy positivo y el negro a* ~ 0.
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    sel = m > 0
    a = lab[:, :, 1][sel].astype(np.int16) - 128
    b = lab[:, :, 2][sel].astype(np.int16) - 128
    n = float(len(a))
    red = a >= col["red_a_min"]
    blue = (b <= col["blue_b_max"]) & ~red
    feats["red_frac"] = float(np.sum(red)) / n
    feats["blue_frac"] = float(np.sum(blue)) / n
    feats["dark_frac"] = float(np.sum(~red & ~blue)) / n
    feats["mean_a"] = float(a.mean())
    feats["n_colors"] = float(sum(feats[k] > 0.15 for k in ("red_frac", "blue_frac", "dark_frac")))
    return feats, m


# ---------------------------------------------------------------------------
# 7. Analisis completo de una escena
# ---------------------------------------------------------------------------
def analyze(img, cfg, debug=None):
    """Ejecuta todo el pipeline. Devuelve dict con dados (centro, cara,
    caracteristicas, clase, traza del arbol) y la jugada interpretada."""
    from tree import classify
    from hand import evaluate_hand

    img, exposure_gain = normalize_exposure(img, cfg)
    img_s, blurred, scale = preprocess(img, cfg)
    dice = detect_dice(blurred, cfg, debug)
    full_conts = [(d["contour"] / scale).astype(np.int32) for d in dice]
    rects = top_face_rects(full_conts, img.shape[:2], cfg)
    # Segunda pasada: conservar las caras originales y recuperar candidatos
    # completos que la perspectiva o el borde del tapete habian descartado.
    # Los candidatos nuevos NO alteran la escala ni los recortes ya aceptados.
    import copy
    recovery = cfg.get("recovery", {})
    if recovery.get("enabled", False):
        # Primero toda la superficie visible; despues la mascara conservadora
        # rescata dados que pudieran fusionarse con el carton al ampliar la zona.
        settings_list = [recovery["segmentation"]]
        fallback = dict(recovery["segmentation"])
        fallback.update(roi_margin_iterations=2, allow_image_edge=False,
                        retain_edge_components=False)
        settings_list.append(fallback)
        for settings in settings_list:
            loose = copy.deepcopy(cfg)
            loose["segmentation"].update(settings)
            extras = []
            for d in detect_dice(blurred, loose):
                matches = [j for j,e in enumerate(dice)
                           if np.linalg.norm(np.array(d["center"])-e["center"]) < 0.6*math.sqrt(e["area"])]
                if matches:
                    j = matches[0]
                    if dice[j].get("recovered", False):
                        contour = (d["contour"] / scale).astype(np.int32)
                        old_score = face_white_score(img, full_conts[j], loose)
                        new_score = face_white_score(img, contour, loose)
                        if new_score > old_score + recovery.get("white_score_margin", .1):
                            d["recovered"] = True
                            dice[j] = d
                            full_conts[j] = contour
                            local = copy.deepcopy(loose)
                            local["perspective"].update(local_side=True, min_dice=1)
                            rects[j] = top_face_rects(full_conts, img.shape[:2], local)[j]
                    continue
                d["recovered"] = True
                extras.append(d)
            if extras:
                all_conts = full_conts + [(d["contour"] / scale).astype(np.int32) for d in extras]
                loose["perspective"]["local_side"] = True
                loose["perspective"]["min_dice"] = 1
                extra_rects = top_face_rects(all_conts, img.shape[:2], loose)[len(dice):]
                dice += extras
                full_conts = all_conts
                rects += extra_rects
    tops = {id(d): r for d, r in zip(dice, rects)}
    others = {id(d): [c for e, c in zip(dice, full_conts) if e is not d] for d in dice}
    results = []
    for d in dice:
        # La deteccion se hace en la imagen reducida, pero la cara se recorta
        # de la foto ORIGINAL: con el tapete entero en la foto el dado mide
        # pocos px y los puntos del 7/8 se perderian al reducir.
        cont = (d["contour"] / scale).astype(np.int32)
        full = {"rect": tops[id(d)], "contour": cont, "others": others[id(d)]}
        face, support = rectify_face(img, full, cfg, with_support=True)
        outward = None
        center_offset = np.array(full['rect'][0]) - np.array([img.shape[1]/2,img.shape[0]/2])
        if cfg.get('perspective',{}).get('color_lateral_enabled',False) and np.linalg.norm(center_offset)>.25*max(img.shape[:2]):
            size=cfg['face']['size_px']
            box=cv2.boxPoints(full['rect']).astype(np.float32)
            dst=np.array([[0,size-1],[0,0],[size-1,0],[size-1,size-1]],np.float32)
            transform=cv2.getPerspectiveTransform(box,dst)
            origin=cv2.perspectiveTransform(np.array([[[img.shape[1]/2,img.shape[0]/2]]],np.float32),transform)[0,0]
            outward=np.array([size/2,size/2])-origin
            outward=outward/max(np.linalg.norm(outward),1e-6)
        feats, sm = extract_features(face, cfg, outward)
        if feats["n_comp"] == 0:
            continue  # mancha clara sin simbolo: no es la cara de un dado
        if feats["ink_ratio"] > cfg["face"].get("ink_ratio_max", 1.0):
            continue  # un bloque de carton/tapete no es tinta sobre plastico blanco
        # La cara de un dado es blanca: L mediana del fondo (sin tinta) medida en
        # fotos reales = 192-253; trozos de fieltro o carton que se cuelan, ~140-150.
        # El blanco artificial del exterior del contorno NO es evidencia de
        # plastico blanco: medir solo fondo real de la cara, sin tinta.
        background = (sm == 0) & support
        face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        Lf = face_lab[:, :, 0][background]
        feats["face_L"] = float(np.median(Lf)) if Lf.size else 0.0
        if feats["face_L"] < cfg["face"]["face_l_min"]:
            continue
        feats["background_S"] = float(np.median(face_hsv[:,:,1][background])) if Lf.size else 255
        feats["background_a"] = float(np.median(face_lab[:,:,1][background])) - 128 if Lf.size else -128
        if (feats["background_S"] > cfg["face"].get("background_s_max", 255) or
                feats["background_a"] < cfg["face"].get("background_a_min", -128)):
            continue
        label, trace = classify(feats, cfg["tree"])
        side_px = math.sqrt(d["area"])
        x, y, w, h = cv2.boundingRect(cont)
        margin = min(x, y, img.shape[1] - x - w, img.shape[0] - y - h)
        notes = []
        if margin < 0.25 * side_px / scale:
            notes.append("Muy cerca del borde: comprobar que la cara superior esta completa")
        if d.get("recovered", False):
            notes.append("Recuperado con filtros ampliados de borde/perspectiva: revisar la cara")
        if feats.get('top_ink_filtered'):
            notes.append('Tinta de lateral separada por color y posicion: revisar la cara superior')
        results.append({
            "center": (d["center"][0] / scale, d["center"][1] / scale),
            "contour": (d["contour"] / scale).astype(np.int32),
            "side_px": side_px / scale,
            "face": face, "symbol_mask": sm,
            "features": feats, "label": label, "trace": trace,
            "recovered": d.get("recovered", False),
            "warnings": notes,
        })
    hand = evaluate_hand([r["label"] for r in results], cfg["classes"])
    mm_per_px = None
    if results:
        mm_per_px = cfg["die"]["side_mm"] / float(np.median([r["side_px"] for r in results]))
    warnings = [f"Dado {i}: {note}" for i, d in enumerate(results, 1) for note in d["warnings"]]
    return {"dice": results, "hand": hand, "mm_per_px": mm_per_px,
            "exposure_gain": exposure_gain,
            "warnings": warnings, "review_required": bool(warnings),
            "image_shape": img.shape[:2]}


# ---------------------------------------------------------------------------
# 8. Visualizacion
# ---------------------------------------------------------------------------
def draw(img, res):
    """Dibuja contornos, numero, clase y jugada sobre una copia de la imagen."""
    out = img.copy()
    t = max(1, img.shape[1] // 800)
    for i, r in enumerate(res["dice"], 1):
        cv2.drawContours(out, [r["contour"]], -1, (0, 255, 255), 2 * t)
        x, y = int(r["center"][0]), int(r["center"][1])
        txt = f"{i}:{r['label']}"
        # Etiqueta a la izquierda del dado y centrada en altura (si no cabe, a la derecha):
        # asi no se confunde con el dado vecino cuando estan en fila.
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, 2 * t)
        off = int(r["side_px"] * 0.65)
        tx = x - off - tw - 6 * t
        if tx < 4:
            tx = x + off + 6 * t
        tx = max(4, min(tx, img.shape[1] - tw - 4))
        org = (tx, max(th + 4, min(y + th // 2, img.shape[0] - 4)))
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, (0, 0, 0), 4 * t)
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, (255, 255, 255), 2 * t)
    banner = f"Dados: {len(res['dice'])} | Jugada: {res['hand']['name']}"
    if res.get("review_required"):
        banner += " | REVISAR BORDES"
    # El rotulo va fuera de la foto: nunca tapa dados del borde superior.
    out = cv2.copyMakeBorder(out, 40 * t, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(out, banner, (10, 30 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * t, (255, 255, 255), 2 * t)
    return out


def face_mosaic(res, cell=128):
    """Mosaico: cara rectificada | mascara del simbolo | clase. Util para el informe."""
    rows = []
    for i, r in enumerate(res["dice"], 1):
        f = cv2.resize(r["face"], (cell, cell))
        m = cv2.cvtColor(cv2.resize(r["symbol_mask"], (cell, cell)), cv2.COLOR_GRAY2BGR)
        lab = np.full((cell, cell, 3), 255, np.uint8)
        cv2.putText(lab, f"{i}:{r['label']}", (8, cell // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        rows.append(np.hstack([f, m, lab]))
    return np.vstack(rows) if rows else np.zeros((cell, 3 * cell, 3), np.uint8)
