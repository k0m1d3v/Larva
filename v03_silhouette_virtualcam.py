"""
Maschera olografica in tempo reale dalla webcam (v3).

Installazione:
    pip install mediapipe opencv-python numpy
    pip install pyvirtualcam        (solo per la webcam virtuale)

Avvio:
    python v03_silhouette_virtualcam.py              finestra normale
    python v03_silhouette_virtualcam.py --virtuale   in piu' manda l'immagine a una webcam virtuale
                                    (Windows/macOS: serve OBS installato;
                                     Linux: serve il modulo v4l2loopback)

Tasti:
    m  cambia stile (ologramma / puntini / linee)
    c  capelli e sagoma accesi/spenti
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
import sys
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
SPESSORE = 1.3        # spessore delle linee, in pixel
BAGLIORE = 0.9        # intensita' dell'alone luminoso
SCIA = 0.72           # persistenza della scia (0 = nessuna, 0.9 = lunghissima)
MAX_SCINTILLE = 900
OGNI_QUANTI = 2       # la sagoma (capelli/corpo) viene ricalcolata ogni N fotogrammi
MODELLO = "face_landmarker.task"
URL_MODELLO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODELLO_SAGOMA = "selfie_multiclass_256x256.tflite"
URL_SAGOMA = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
STILI = ["ologramma", "puntini", "linee"]

# segmenti di occhi, sopracciglia, labbra e ovale: rendono leggibili le espressioni
_C = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
CONTORNI = np.array([[c.start, c.end] for c in _C], np.int32)
BOCCA = np.unique(
    [[c.start, c.end] for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS]
)



def _triangoli():
    """Ricava i triangoli della mesh dai suoi spigoli (terne di punti tutti collegati)."""
    vicini = {}
    for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        vicini.setdefault(c.start, set()).add(c.end)
        vicini.setdefault(c.end, set()).add(c.start)
    tri = [(a, b, c) for a in vicini for b in vicini[a] if b > a
           for c in vicini[a] & vicini[b] if c > b]
    return np.array(tri, np.int32)


TRIANGOLI = _triangoli()

rng = np.random.default_rng()


def scarica(nome, url):
    if not os.path.exists(nome):
        print(f"Scarico {nome} (solo la prima volta)...")
        urllib.request.urlretrieve(url, nome)


def crea_landmarker():
    """Scarica il modello al primo avvio e crea il tracciatore del volto."""
    scarica(MODELLO, URL_MODELLO)
    opzioni = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELLO),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,      # punteggi delle espressioni
    )
    return vision.FaceLandmarker.create_from_options(opzioni)


def crea_segmentatore():
    """Modello che separa capelli, corpo e sfondo. Se non parte, si va avanti senza."""
    try:
        scarica(MODELLO_SAGOMA, URL_SAGOMA)
        opzioni = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELLO_SAGOMA),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
        )
        return vision.ImageSegmenter.create_from_options(opzioni)
    except Exception as errore:
        print("Sagoma disattivata:", errore)
        return None


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
        self.capelli = None               # maschere morbide 0..1 grandi come il fotogramma
        self.corpo = None

    def imposta_sagoma(self, categorie, dimensione):
        """categorie: mappa del segmentatore (1 = capelli, 2 = pelle del corpo, 4 = vestiti)."""
        def morbida(maschera, vecchia):
            m = cv2.resize(maschera.astype(np.float32), dimensione, interpolation=cv2.INTER_LINEAR)
            m = cv2.GaussianBlur(m, (0, 0), 3)
            return m if vecchia is None or vecchia.shape != m.shape else vecchia * 0.5 + m * 0.5
        self.capelli = morbida(categorie == 1, self.capelli)
        self.corpo = morbida((categorie == 2) | (categorie == 4), self.corpo)

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

        # profondita' vera: riempio ogni triangolo della mesh, dal piu' lontano al piu' vicino
        zn = (z - z.min()) / (np.ptp(z) + 1e-6)
        tri = (p - (x0, y0)).astype(np.int32)[TRIANGOLI]          # (T, 3, 2)
        z_tri = zn[TRIANGOLI].mean(axis=1)
        profondita = np.full((y1 - y0, x1 - x0), -1.0, np.float32)
        for i in np.argsort(-z_tri):
            cv2.fillConvexPoly(profondita, tri[i], float(z_tri[i]))
        maschera = (profondita >= 0).astype(np.float32)
        sigma = max(1.0, larghezza / 70.0)                        # smussa le sfaccettature
        profondita = cv2.GaussianBlur(profondita * maschera, (0, 0), sigma) / (
            cv2.GaussianBlur(maschera, (0, 0), sigma) + 1e-6)

        # linee di livello di spessore costante: distanza in pixel dalla curva piu' vicina
        livello = N_LINEE * np.sqrt(profondita) - self.fase       # sqrt: piu' linee sul naso, meno sui lati
        scarto = np.abs(livello - np.round(livello))
        gx = cv2.Sobel(livello, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(livello, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        distanza = scarto / (np.sqrt(gx * gx + gy * gy) + 0.02)
        linee = np.exp(-((distanza / SPESSORE) ** 2))

        # fascio di scansione che scende lungo il volto
        righe = np.arange(y1 - y0, dtype=np.float32)[:, None]
        y_fascio = ((t * 0.45) % 1.3 - 0.15) * (y1 - y0)
        fascio = np.exp(-(((righe - y_fascio) / (larghezza / 30.0)) ** 2))
        linee = linee * (0.6 + 1.5 * fascio) + 0.2 * fascio

        maschera = cv2.GaussianBlur(maschera, (0, 0), 1.5)       # bordo morbido
        linee *= maschera * self.presenza

        zona = strato[y0:y1, x0:x1].astype(np.float32) + linee[..., None] * colore
        strato[y0:y1, x0:x1] = np.clip(zona, 0, 255).astype(np.uint8)

    def _sagoma(self, strato, colore, t):
        """Capelli a strisce verticali che scorrono, corpo a righe orizzontali tenui."""
        h, w = strato.shape[:2]
        if self.capelli is None or self.capelli.shape != (h, w):
            return
        xx = np.arange(w, dtype=np.float32)[None, :]
        yy = np.arange(h, dtype=np.float32)[:, None]
        passo = max(4.0, w / 110.0)
        ondeggio = 3.0 * np.sin(yy / 23.0 + t * 1.5)
        strisce = (0.5 + 0.5 * np.cos(2 * np.pi * (xx + ondeggio) / passo)) ** 4
        flusso = 0.6 + 0.4 * np.sin(yy / 15.0 - t * 4.0)          # luce che cola verso il basso
        righe = (0.5 + 0.5 * np.cos(2 * np.pi * yy / (passo * 1.5) - t * 2.0)) ** 6
        luce = self.capelli * strisce * flusso * 0.8 + self.corpo * righe * 0.3
        somma = strato.astype(np.float32) + (luce * self.presenza)[..., None] * colore
        strato[:] = np.clip(somma, 0, 255).astype(np.uint8)

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

    def disegna(self, fotogramma, stile, sfondo_nero, glitch, t, sagoma=True):
        h, w = fotogramma.shape[:2]
        colore = colore_da_sorriso(self.sorriso)
        strato = np.zeros_like(fotogramma)

        if self.pos is not None and self.presenza > 0:
            if sagoma and stile == "ologramma":
                self._sagoma(strato, colore, t)
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
    segmentatore = crea_segmentatore()
    cam = cv2.VideoCapture(CAMERA)
    if not cam.isOpened():
        raise SystemExit("Non riesco ad aprire la webcam: prova a cambiare CAMERA.")

    virtuale = None
    if "--virtuale" in sys.argv:
        import pyvirtualcam
        larg = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        alt = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        virtuale = pyvirtualcam.Camera(width=larg, height=alt, fps=30)
        print("Webcam virtuale attiva:", virtuale.device)

    olo = Ologramma()
    stile, sfondo_nero, glitch, sagoma = 0, True, True, True
    n = 0
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

        if segmentatore and sagoma and n % OGNI_QUANTI == 0:
            categorie = segmentatore.segment_for_video(immagine, int(t * 1000)).category_mask
            olo.imposta_sagoma(categorie.numpy_view(), (w, h))
        n += 1

        olo.aggiorna(punti, espressioni, dt)
        uscita = olo.disegna(fotogramma, STILI[stile], sfondo_nero, glitch, t, sagoma)
        cv2.imshow("Maschera", uscita)
        if virtuale:
            virtuale.send(cv2.cvtColor(uscita, cv2.COLOR_BGR2RGB))

        tasto = cv2.waitKey(1) & 0xFF
        if tasto == ord("q"):
            break
        if tasto == ord("m"):
            stile = (stile + 1) % len(STILI)
        if tasto == ord("c"):
            sagoma = not sagoma
        if tasto == ord("g"):
            glitch = not glitch
        if tasto == ord("b"):
            sfondo_nero = not sfondo_nero

    cam.release()
    if virtuale:
        virtuale.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()