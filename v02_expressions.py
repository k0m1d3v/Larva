"""
Maschera olografica in tempo reale dalla webcam (v2).

Installazione:
    pip install mediapipe opencv-python numpy

Avvio:
    python v02_expressions.py

Tasti:
    m  cambia stile (ologramma / puntini / linee)
    g  glitch acceso/spento
    b  sfondo nero <-> webcam scurita
    q  esci

Reazioni alle espressioni:
    bocca aperta  -> la maschera pulsa e sputa scintille
    sorriso       -> il colore vira dal ciano al magenta
    sopracciglia  -> le linee scorrono piu' veloci
    volto perso   -> le particelle si disperdono, poi si ricompongono
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------- parametri
CAMERA = 0            # indice della webcam
N_LINEE = 16          # quante linee di livello sul volto
BAGLIORE = 0.9        # intensita' dell'alone luminoso
SCIA = 0.72           # persistenza della scia (0 = nessuna, 0.9 = lunghissima)
MAX_SCINTILLE = 900
MODELLO = "face_landmarker.task"
URL_MODELLO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
STILI = ["ologramma", "puntini", "linee"]

# segmenti di occhi, sopracciglia, labbra e ovale: rendono leggibili le espressioni
_C = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
CONTORNI = np.array([[c.start, c.end] for c in _C], np.int32)
BOCCA = np.unique(
    [[c.start, c.end] for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS]
)

rng = np.random.default_rng()


def crea_landmarker():
    """Scarica il modello al primo avvio e crea il tracciatore del volto."""
    if not os.path.exists(MODELLO):
        print("Scarico il modello del volto (solo la prima volta)...")
        urllib.request.urlretrieve(URL_MODELLO, MODELLO)
    opzioni = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELLO),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,      # punteggi delle espressioni
    )
    return vision.FaceLandmarker.create_from_options(opzioni)


def colore_da_sorriso(sorriso):
    """Ciano a riposo, magenta quando sorridi. Restituisce un colore BGR."""
    tinta = 95 + 60 * float(np.clip(sorriso, 0, 1))          # scala tinta di OpenCV
    hsv = np.uint8([[[tinta, 215, 255]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


class Ologramma:
    """Tiene lo stato tra un fotogramma e l'altro: particelle, scintille, scia."""

    def __init__(self):
        self.pos = None                   # posizione attuale delle particelle (N, 2)
        self.vel = None
        self.z = None
        self.presenza = 0.0               # 0 = volto assente, 1 = volto agganciato
        self.sc_pos = np.zeros((0, 2), np.float32)   # scintille
        self.sc_vel = np.zeros((0, 2), np.float32)
        self.sc_vita = np.zeros(0, np.float32)
        self.scia = None
        self.energia = 0.0                # bocca aperta, smussata nel tempo
        self.sorriso = 0.0
        self.sopracciglia = 0.0
        self.fase = 0.0                   # scorrimento delle linee

    # ------------------------------------------------------------ fisica
    def aggiorna(self, punti, espressioni, dt):
        e = espressioni or {}
        liscia = lambda vecchio, nuovo: vecchio + 0.25 * (nuovo - vecchio)
        self.energia = liscia(self.energia, e.get("jawOpen", 0.0))
        self.sorriso = liscia(
            self.sorriso, (e.get("mouthSmileLeft", 0.0) + e.get("mouthSmileRight", 0.0)) / 2
        )
        self.sopracciglia = liscia(self.sopracciglia, e.get("browInnerUp", 0.0))
        self.fase += dt * (0.25 + 1.2 * self.sopracciglia + 0.8 * self.energia)

        if punti is not None:
            if self.pos is None:
                self.pos = punti[:, :2].copy()
                self.vel = np.zeros_like(self.pos)
            # molla smorzata: le particelle inseguono il volto con un po' di inerzia
            self.vel = self.vel * 0.55 + (punti[:, :2] - self.pos) * 0.38
            self.pos += self.vel
            self.z = punti[:, 2]
            self.presenza = min(1.0, self.presenza + 0.12)
            if self.energia > 0.25:
                self._emetti(punti)
        elif self.pos is not None:
            # volto perso: le particelle si sciolgono verso l'alto
            self.vel = self.vel * 0.97 + rng.normal(0, 0.6, self.vel.shape).astype(np.float32)
            self.vel[:, 1] -= 0.25
            self.pos += self.vel
            self.presenza = max(0.0, self.presenza - 0.03)

        if len(self.sc_vita):
            self.sc_pos += self.sc_vel
            self.sc_vel *= 0.96
            self.sc_vita -= dt * 1.4
            vive = self.sc_vita > 0
            self.sc_pos, self.sc_vel, self.sc_vita = (
                self.sc_pos[vive], self.sc_vel[vive], self.sc_vita[vive])

    def _emetti(self, punti):
        """Scintille che escono dalla bocca, proporzionali a quanto e' aperta."""
        n = int(40 * self.energia)
        bocca = punti[BOCCA, :2]
        centro = bocca.mean(axis=0)
        origine = bocca[rng.integers(0, len(bocca), n)]
        direzione = origine - centro
        direzione /= np.linalg.norm(direzione, axis=1, keepdims=True) + 1e-6
        scala = np.ptp(punti[:, 0]) / 60.0
        vel = direzione * rng.uniform(1.0, 4.0, (n, 1)) * scala + rng.normal(0, 0.6, (n, 2))
        self.sc_pos = np.vstack([self.sc_pos, origine])[-MAX_SCINTILLE:].astype(np.float32)
        self.sc_vel = np.vstack([self.sc_vel, vel])[-MAX_SCINTILLE:].astype(np.float32)
        self.sc_vita = np.concatenate(
            [self.sc_vita, rng.uniform(0.5, 1.0, n)])[-MAX_SCINTILLE:].astype(np.float32)

    # ------------------------------------------------------------ disegno
    def _linee(self, strato, colore, t):
        """Linee di livello animate sulla profondita' del volto, con fascio di scansione."""
        h, w = strato.shape[:2]
        p, z = self.pos, self.z
        larghezza = np.ptp(p[:, 0]) + 1e-6
        x0, y0 = np.floor(p.min(axis=0)).astype(int) - 4
        x1, y1 = np.ceil(p.max(axis=0)).astype(int) + 4
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return

        dentro = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
        px = (p[dentro, 0] - x0).astype(int)
        py = (p[dentro, 1] - y0).astype(int)
        if len(px) < 10:
            return
        zn = (z[dentro] - z.min()) / (np.ptp(z) + 1e-6)

        # mappa di profondita' densa: "spruzzo" i punti e li sfumo
        somma = np.zeros((y1 - y0, x1 - x0), np.float32)
        peso = np.zeros_like(somma)
        np.add.at(somma, (py, px), zn)
        np.add.at(peso, (py, px), 1.0)
        sigma = larghezza / 22.0
        profondita = cv2.GaussianBlur(somma, (0, 0), sigma) / (
            cv2.GaussianBlur(peso, (0, 0), sigma) + 1e-6)

        onda = 0.5 + 0.5 * np.cos(2 * np.pi * (N_LINEE * profondita - self.fase))
        linee = onda ** 6

        # fascio di scansione che scende lungo il volto
        righe = np.arange(y1 - y0, dtype=np.float32)[:, None]
        y_fascio = ((t * 0.45) % 1.3 - 0.15) * (y1 - y0)
        fascio = np.exp(-(((righe - y_fascio) / (larghezza / 30.0)) ** 2))
        linee = linee * (0.6 + 1.5 * fascio) + 0.2 * fascio

        contorno = cv2.convexHull(np.stack([px, py], axis=1).astype(np.int32))
        maschera = np.zeros_like(somma)
        cv2.fillConvexPoly(maschera, contorno, 1.0, cv2.LINE_AA)
        maschera = cv2.GaussianBlur(maschera, (0, 0), max(1.0, sigma / 3))
        linee *= maschera * self.presenza

        zona = strato[y0:y1, x0:x1].astype(np.float32) + linee[..., None] * colore
        strato[y0:y1, x0:x1] = np.clip(zona, 0, 255).astype(np.uint8)

    def _puntini(self, strato, colore):
        z = self.z
        vicinanza = 1.0 - (z - z.min()) / (np.ptp(z) + 1e-6)
        scala = max(np.ptp(self.pos[:, 0]), 60.0) / 250.0 * (1.0 + 0.6 * self.energia)
        for (x, y), v in zip(self.pos, vicinanza):
            lum = float((0.4 + 0.6 * v) * self.presenza)
            cv2.circle(strato, (int(x), int(y)), max(1, int(round((1 + 2 * v) * scala))),
                       tuple(float(c) * lum for c in colore), -1, cv2.LINE_AA)

    def _contorni(self, strato, colore):
        segmenti = self.pos[CONTORNI].astype(np.int32)           # (M, 2, 2)
        chiaro = tuple(float(c) for c in (colore * 0.6 + 100) * self.presenza)
        cv2.polylines(strato, list(segmenti), False, chiaro, 1, cv2.LINE_AA)

    def _scintille(self, strato, colore):
        h, w = strato.shape[:2]
        for (x, y), vita in zip(self.sc_pos, self.sc_vita):
            if 0 <= x < w and 0 <= y < h:
                c = tuple(float(v) for v in (colore * 0.5 + 127) * min(1.0, vita * 1.5))
                cv2.circle(strato, (int(x), int(y)), 1, c, -1, cv2.LINE_AA)

    def disegna(self, fotogramma, stile, sfondo_nero, glitch, t):
        h, w = fotogramma.shape[:2]
        colore = colore_da_sorriso(self.sorriso)
        strato = np.zeros_like(fotogramma)

        if self.pos is not None and self.presenza > 0:
            if stile in ("ologramma", "linee") and self.presenza > 0.5:
                self._linee(strato, colore, t)
            if stile in ("ologramma", "puntini"):
                self._puntini(strato, colore)
            if stile == "ologramma" and self.presenza > 0.9:
                self._contorni(strato, colore)
        self._scintille(strato, colore)

        # scia: il fotogramma precedente svanisce piano invece di sparire
        if self.scia is None or self.scia.shape != strato.shape:
            self.scia = strato.copy()
        else:
            self.scia = cv2.max(strato, cv2.convertScaleAbs(self.scia, alpha=SCIA))
        img = self.scia

        # alone su due scale (calcolato a bassa risoluzione: e' veloce)
        piccolo = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        alone = cv2.addWeighted(cv2.GaussianBlur(piccolo, (0, 0), 2), 0.6,
                                cv2.GaussianBlur(piccolo, (0, 0), 8), 0.6, 0)
        alone = cv2.resize(alone, (w, h), interpolation=cv2.INTER_LINEAR)
        forza = BAGLIORE * (1.0 + 0.5 * self.energia)
        img = cv2.add(img, cv2.convertScaleAbs(alone, alpha=forza))

        # aberrazione cromatica: canali leggermente sfalsati
        d = 2 + int(6 * self.energia)
        img = cv2.merge([np.roll(img[..., 0], -d, 1), img[..., 1], np.roll(img[..., 2], d, 1)])

        if glitch and rng.random() < 0.04 + 0.25 * self.energia:
            for _ in range(rng.integers(2, 6)):                  # bande orizzontali spostate
                y = rng.integers(0, h - 4)
                alt = rng.integers(3, max(4, h // 12))
                img[y:y + alt] = np.roll(img[y:y + alt], int(rng.integers(-40, 40)), axis=1)

        img[1::3] = (img[1::3] * 0.7).astype(np.uint8)           # righe di scansione
        img = cv2.convertScaleAbs(img, alpha=float(0.9 + 0.1 * rng.random()))   # sfarfallio

        base = np.zeros_like(fotogramma) if sfondo_nero else (fotogramma * 0.25).astype(np.uint8)
        return cv2.add(base, img)


def main():
    landmarker = crea_landmarker()
    cam = cv2.VideoCapture(CAMERA)
    if not cam.isOpened():
        raise SystemExit("Non riesco ad aprire la webcam: prova a cambiare CAMERA.")

    olo = Ologramma()
    stile, sfondo_nero, glitch = 0, True, True
    inizio = prima = time.monotonic()
    while True:
        ok, fotogramma = cam.read()
        if not ok:
            break
        fotogramma = cv2.flip(fotogramma, 1)                      # effetto specchio
        h, w = fotogramma.shape[:2]
        adesso = time.monotonic()
        t, dt, prima = adesso - inizio, min(adesso - prima, 0.1), adesso

        rgb = cv2.cvtColor(fotogramma, cv2.COLOR_BGR2RGB)
        immagine = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        risultato = landmarker.detect_for_video(immagine, int(t * 1000))

        punti, espressioni = None, None
        if risultato.face_landmarks:
            volto = risultato.face_landmarks[0]
            # x, y in pixel; z (profondita') nella stessa scala di x
            punti = np.array([[p.x * w, p.y * h, p.z * w] for p in volto], np.float32)
            if risultato.face_blendshapes:
                espressioni = {b.category_name: b.score for b in risultato.face_blendshapes[0]}

        olo.aggiorna(punti, espressioni, dt)
        cv2.imshow("Maschera", olo.disegna(fotogramma, STILI[stile], sfondo_nero, glitch, t))

        tasto = cv2.waitKey(1) & 0xFF
        if tasto == ord("q"):
            break
        if tasto == ord("m"):
            stile = (stile + 1) % len(STILI)
        if tasto == ord("g"):
            glitch = not glitch
        if tasto == ord("b"):
            sfondo_nero = not sfondo_nero

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()