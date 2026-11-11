"""
Maschera olografica in tempo reale dalla webcam.

Installazione:
    pip install mediapipe opencv-python numpy

Avvio:
    python v01_prototype.py

Tasti:
    m  cambia stile (puntini / linee di livello / entrambi)
    b  sfondo nero <-> webcam scurita
    q  esci
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
CAMERA = 0                    # indice della webcam
COLORE = (255, 210, 40)       # BGR: azzurro/ciano
N_LINEE = 14                  # quante linee di livello sul volto
BAGLIORE = 1.6                # intensita' dell'alone luminoso
MODELLO = "face_landmarker.task"
URL_MODELLO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
STILI = ["puntini", "linee", "entrambi"]


def crea_landmarker():
    """Scarica il modello al primo avvio e crea il tracciatore del volto."""
    if not os.path.exists(MODELLO):
        print("Scarico il modello del volto (solo la prima volta)...")
        urllib.request.urlretrieve(URL_MODELLO, MODELLO)
    opzioni = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELLO),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opzioni)


def disegna_puntini(strato, punti):
    """Un puntino per ogni landmark: piu' vicino alla camera = piu' grande e luminoso."""
    z = punti[:, 2]
    vicinanza = 1.0 - (z - z.min()) / (np.ptp(z) + 1e-6)   # 0 lontano, 1 vicino
    scala = np.ptp(punti[:, 0]) / 250.0                     # adatta alla distanza dal volto
    for (x, y, _), v in zip(punti, vicinanza):
        raggio = max(1, int(round((1.0 + 2.0 * v) * scala)))
        lum = 0.35 + 0.65 * v
        colore = tuple(int(c * lum) for c in COLORE)
        cv2.circle(strato, (int(x), int(y)), raggio, colore, -1, cv2.LINE_AA)


def disegna_linee(strato, punti):
    """Linee di livello sulla profondita' del volto (effetto 'topografico')."""
    h, w = strato.shape[:2]
    larghezza_volto = np.ptp(punti[:, 0]) + 1e-6

    # riquadro attorno al volto, per non elaborare tutto il fotogramma
    x0, y0 = np.floor(punti[:, :2].min(axis=0)).astype(int) - 4
    x1, y1 = np.ceil(punti[:, :2].max(axis=0)).astype(int) + 4
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return

    # mappa di profondita' densa: "spruzzo" i punti e li sfumo
    px = np.clip(punti[:, 0].astype(int) - x0, 0, x1 - x0 - 1)
    py = np.clip(punti[:, 1].astype(int) - y0, 0, y1 - y0 - 1)
    zn = (punti[:, 2] - punti[:, 2].min()) / larghezza_volto
    somma = np.zeros((y1 - y0, x1 - x0), np.float32)
    peso = np.zeros_like(somma)
    np.add.at(somma, (py, px), zn)
    np.add.at(peso, (py, px), 1.0)
    sigma = larghezza_volto / 22.0
    somma = cv2.GaussianBlur(somma, (0, 0), sigma)
    peso = cv2.GaussianBlur(peso, (0, 0), sigma)
    profondita = somma / (peso + 1e-6)

    # strisce periodiche sulla profondita' -> linee di livello
    onda = 0.5 + 0.5 * np.cos(2 * np.pi * N_LINEE * profondita / (zn.max() + 1e-6))
    linee = onda ** 6

    # tengo solo l'interno del volto, con bordo morbido
    contorno = cv2.convexHull(np.stack([px, py], axis=1).astype(np.int32))
    maschera = np.zeros_like(somma)
    cv2.fillConvexPoly(maschera, contorno, 1.0, cv2.LINE_AA)
    maschera = cv2.GaussianBlur(maschera, (0, 0), max(1.0, sigma / 3))
    linee *= maschera

    zona = strato[y0:y1, x0:x1].astype(np.float32)
    zona += linee[..., None] * np.array(COLORE, np.float32)
    strato[y0:y1, x0:x1] = np.clip(zona, 0, 255).astype(np.uint8)


def componi(fotogramma, punti, stile, sfondo_nero):
    """Costruisce l'immagine finale: sfondo + maschera + alone luminoso."""
    strato = np.zeros_like(fotogramma)
    if punti is not None:
        if stile in ("linee", "entrambi"):
            disegna_linee(strato, punti)
        if stile in ("puntini", "entrambi"):
            disegna_puntini(strato, punti)

    alone = cv2.GaussianBlur(strato, (0, 0), 9)
    base = np.zeros_like(fotogramma) if sfondo_nero else (fotogramma * 0.25).astype(np.uint8)
    uscita = cv2.add(base, strato)
    return cv2.add(uscita, cv2.convertScaleAbs(alone, alpha=BAGLIORE))


def main():
    landmarker = crea_landmarker()
    cam = cv2.VideoCapture(CAMERA)
    if not cam.isOpened():
        raise SystemExit("Non riesco ad aprire la webcam: prova a cambiare CAMERA.")

    stile, sfondo_nero = 0, True
    inizio = time.monotonic()
    while True:
        ok, fotogramma = cam.read()
        if not ok:
            break
        fotogramma = cv2.flip(fotogramma, 1)                      # effetto specchio
        h, w = fotogramma.shape[:2]

        rgb = cv2.cvtColor(fotogramma, cv2.COLOR_BGR2RGB)
        immagine = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_ms = int((time.monotonic() - inizio) * 1000)
        risultato = landmarker.detect_for_video(immagine, t_ms)

        punti = None
        if risultato.face_landmarks:
            volto = risultato.face_landmarks[0]
            # x, y in pixel; z (profondita') nella stessa scala di x
            punti = np.array([[p.x * w, p.y * h, p.z * w] for p in volto], np.float32)

        cv2.imshow("Maschera", componi(fotogramma, punti, STILI[stile], sfondo_nero))

        tasto = cv2.waitKey(1) & 0xFF
        if tasto == ord("q"):
            break
        if tasto == ord("m"):
            stile = (stile + 1) % len(STILI)
        if tasto == ord("b"):
            sfondo_nero = not sfondo_nero

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()