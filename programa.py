"""
programa.py - Programa principal del Reto 07 (dados de poker).

Se ejecuta, se elige una foto cenital del tapete y muestra:
    - Nivel 1: los dados detectados (contorno y numero)
    - Nivel 2: la cara de cada dado y las decisiones del arbol que llevaron a ella
    - Nivel 3: la jugada (repoker, poker, full, escalera, trio, doble pareja, pareja, nada)
Ademas permite ver la mascara de segmentacion y las caras rectificadas, y guarda
las figuras en salida/ para el informe.

Uso:
    py programa.py              # abre la ventana; boton "Abrir foto..."
    py programa.py foto.jpg     # abre la ventana con esa foto ya analizada
"""
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import vision

OUT_DIR = os.path.join(vision.HERE, "salida")
VIEW_W, VIEW_H = 900, 680  # tamano maximo de la imagen en pantalla


class App:
    def __init__(self, root):
        self.root = root
        self.cfg = vision.load_config()
        self.views = {}      # nombre de vista -> imagen BGR
        self.photo = None    # referencia a PhotoImage (si no, tkinter la libera)
        root.title("Reto 07 - Dados de poker (visión artificial clásica)")
        root.geometry("1280x760")

        # Barra superior
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Abrir foto...", command=self.open_dialog).pack(side="left")
        self.file_lbl = ttk.Label(top, text="Ninguna foto cargada")
        self.file_lbl.pack(side="left", padx=12)
        self.view_var = tk.StringVar(value="Resultado")
        for name in ("Caras", "Máscara", "Resultado"):
            ttk.Radiobutton(top, text=name, value=name, variable=self.view_var,
                            command=self.show_view).pack(side="right", padx=4)
        ttk.Label(top, text="Ver:").pack(side="right")

        body = ttk.Frame(root, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)

        # Panel de resultados (derecha)
        side = ttk.Frame(body, width=340, padding=(12, 0, 0, 0))
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="Jugada", font=("Segoe UI", 11)).pack(anchor="w")
        self.hand_lbl = ttk.Label(side, text="-", font=("Segoe UI", 26, "bold"))
        self.hand_lbl.pack(anchor="w")
        self.detail_lbl = ttk.Label(side, text="", wraplength=320, font=("Segoe UI", 10))
        self.detail_lbl.pack(anchor="w", pady=(0, 10))
        ttk.Label(side, text="Dados detectados", font=("Segoe UI", 11)).pack(anchor="w")
        self.dice_txt = tk.Text(side, height=8, width=40, font=("Consolas", 11), relief="flat")
        self.dice_txt.pack(fill="x", pady=(2, 10))
        ttk.Label(side, text="Decisiones del árbol (por dado)", font=("Segoe UI", 11)).pack(anchor="w")
        self.tree_txt = tk.Text(side, height=14, width=40, font=("Consolas", 9), relief="flat", wrap="word")
        self.tree_txt.pack(fill="both", expand=True, pady=(2, 0))

        # Imagen (izquierda)
        self.img_lbl = ttk.Label(body, text="Pulsa \"Abrir foto...\" y elige una foto del tapete",
                                 anchor="center")
        self.img_lbl.pack(side="left", fill="both", expand=True)

        self.status = ttk.Label(root, text="", padding=(8, 2), relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    # -- acciones ----------------------------------------------------------
    def open_dialog(self):
        path = filedialog.askopenfilename(
            title="Elige una foto del tapete",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")])
        if path:
            self.load(path)

    def load(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"No se puede leer la imagen:\n{path}")
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        t0 = time.time()
        dbg = {}
        res = vision.analyze(img, self.cfg, dbg)
        dt = time.time() - t0
        self.root.config(cursor="")

        self.views = {"Resultado": vision.draw(img, res),
                      "Máscara": cv2.cvtColor(dbg["object_mask"], cv2.COLOR_GRAY2BGR),
                      "Caras": vision.face_mosaic(res)}
        self.file_lbl.config(text=os.path.basename(path))
        self.fill_panel(res)
        self.show_view()

        # Guardar figuras para el informe
        os.makedirs(OUT_DIR, exist_ok=True)
        base = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(path))[0])
        cv2.imwrite(base + "_resultado.png", self.views["Resultado"])
        cv2.imwrite(base + "_mascara.png", dbg["object_mask"])
        cv2.imwrite(base + "_caras.png", self.views["Caras"])
        escala = f" · escala {res['mm_per_px']:.3f} mm/px" if res["mm_per_px"] else ""
        if res.get("exposure_gain", 1) > 1.05:
            escala += " · luz compensada automáticamente"
        self.status.config(text=f"Analizada en {dt:.2f} s{escala} · figuras guardadas en salida/")
        return res

    def fill_panel(self, res):
        h = res["hand"]
        self.hand_lbl.config(text=h["name"] + (" *" if res.get("review_required") else ""))
        det = h["detail"]
        if not h["complete"] and res["dice"]:
            det += f"   (hay {len(res['dice'])} dados, no 5)"
        if res.get("review_required"):
            det += "\n* Lectura provisional: revisa los dados marcados y la vista Caras."
        self.detail_lbl.config(text=det)
        self.dice_txt.delete("1.0", "end")
        self.tree_txt.delete("1.0", "end")
        if not res["dice"]:
            self.dice_txt.insert("end", "No se ha detectado ningún dado.")
            return
        for i, d in enumerate(res["dice"], 1):
            self.dice_txt.insert("end", f"Dado {i}:  {d['label']}" + (" [revisar]" if d.get("warnings") else "") + "\n")
            self.tree_txt.insert("end", f"Dado {i} -> {d['label']}\n")
            for warning in d.get("warnings", []):
                self.tree_txt.insert("end", f"   AVISO: {warning}\n")
            for step in d["trace"]:
                self.tree_txt.insert("end", f"   {step}\n")

    def show_view(self):
        img = self.views.get(self.view_var.get())
        if img is None:
            return
        h, w = img.shape[:2]
        # La foto se reduce para caber; el mosaico de caras (pequeno) se puede ampliar
        s = min(VIEW_W / w, VIEW_H / h, 3.0 if self.view_var.get() == "Caras" else 1.0)
        disp = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        self.img_lbl.config(image=self.photo, text="")


def main():
    # Windows con escalado de pantalla (125 %, 150 %...): sin esto la ventana sale
    # borrosa y mas grande que la pantalla, y se corta el panel de la derecha.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = App(root)
    if len(sys.argv) > 1:
        root.after(100, lambda: app.load(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
