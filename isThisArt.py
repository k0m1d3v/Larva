"""
Maschera olografica in tempo reale dalla webcam (v8, rendering su GPU).

Installazione:
    pip install mediapipe opencv-python numpy moderngl glfw
    pip install sounddevice         (facoltativo: reazione alla voce)
    pip install pyvirtualcam        (facoltativo: webcam virtuale)

Avvio:
    python maschera.py              finestra normale
    python maschera.py --virtuale   in piu' manda l'immagine a una webcam virtuale
                                    (Windows/macOS: serve OBS installato;
                                     Linux: serve il modulo v4l2loopback)

Tasti:
    TAB  pannello con i cursori (linee, bagliore, scia, prospettiva, colore...)
    1-4  preset: ciano, rosso allarme, oro, bianco e nero
    l    motivo delle linee: curve di livello / anelli attorno a occhi e bocca / scansione
    m    stile: ologramma / solo puntini / solo linee
    z    volto centrato, ingrandito e in prospettiva <-> posizione reale
    o    iridi            d  denti            t  tratti (sopracciglia, occhi, labbra)
    k    corna            a  ali da demone in caratteri ASCII (misure dal pannello)
    g    glitch           b  sfondo nero <-> webcam scurita
    p    selettore di colore di sistema
    frecce sinistra/destra  tinta          frecce su/giu'  saturazione
    f    schermo intero
    s    salva uno screenshot (PNG)       r  avvia/ferma la registrazione video (MP4)
    q    esci

Le impostazioni vengono salvate in maschera_impostazioni.json alla chiusura
e ricaricate all'avvio (cancella il file per tornare ai valori iniziali).

Reazioni:
    voce                    -> un'onda di luce parte dalla bocca a ogni sillaba
    sopracciglia aggrottate -> la maschera diventa rossa e trema
    sorpresa (occhi sgranati) -> lampo bianco
    occhiolino              -> glitch secco
    sopracciglia alzate     -> le linee scorrono piu' veloci
    entri/esci dall'inquadratura -> il volto si materializza/dissolve dal naso in fuori
"""

import colorsys
import json
import os
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import moderngl
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------- parametri
CAMERA = 0            # indice della webcam
ALTEZZA = 0           # risoluzione verticale del rendering; 0 = quella del monitor (max 1080)
SOVRACAMPIONA = 2     # la scena e' disegnata a risoluzione x2 e poi ridotta (1 = spento)
N_ALONI = 6           # scale del bagliore: piu' sono, piu' l'alone e' largo e morbido
GRANDEZZA = 0.64      # con 'z' attivo: altezza del volto rispetto alla finestra
PROSPETTIVA = 0.9     # valore iniziale. 0 = proiezione piatta; piu' alto = piu' profondita'
OSCILLAZIONE = 0.10   # ampiezza (radianti) del dondolio della camera virtuale
SCIA = 0.80           # persistenza della scia (0 = nessuna, 0.95 = lunghissima)
MODELLO = "face_landmarker.task"
URL_MODELLO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FILE_IMPOSTAZIONI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "maschera_impostazioni.json")
MODELLO_CORPO = "pose_landmarker_lite.task"
URL_CORPO = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
STILI = ["ologramma", "puntini", "linee"]
MOTIVI = ["livello", "anelli", "scansione"]
#          nome             tinta  satur.  linee  bagliore  spessore
PRESET = {
    1: ("ciano",            0.50,  0.85,   30,    1.00,     1.1),
    2: ("rosso allarme",    0.99,  0.90,   36,    1.25,     1.1),
    3: ("oro",              0.12,  0.80,   26,    1.10,     1.2),
    4: ("bianco e nero",    0.00,  0.00,   22,    0.45,     1.6),
}

_FLC = vision.FaceLandmarksConnections


def _triangoli():
    """Ricava i triangoli della mesh dai suoi spigoli (terne di punti tutti collegati)."""
    vicini = {}
    for c in _FLC.FACE_LANDMARKS_TESSELATION:
        vicini.setdefault(c.start, set()).add(c.end)
        vicini.setdefault(c.end, set()).add(c.start)
    tri = [(a, b, c) for a in vicini for b in vicini[a] if b > a
           for c in vicini[a] & vicini[b] if c > b]
    return np.array(tri, np.int32)


def _suddividi(tri, primo):
    """Divide ogni triangolo in quattro: restituisce gli spigoli e i nuovi triangoli.
    primo: indice del primo vertice nuovo (i punti medi vengono accodati agli esistenti)."""
    spigoli = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1), axis=0)
    indice = {tuple(s): primo + i for i, s in enumerate(spigoli)}
    meta = lambda a, b: indice[(min(a, b), max(a, b))]
    nuovi = []
    for a, b, c in tri:
        ab, bc, ca = meta(a, b), meta(b, c), meta(c, a)
        nuovi += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    return spigoli.astype(np.int32), np.array(nuovi, np.int32)


TRIANGOLI = _triangoli()
# due suddivisioni: ogni triangolo della mesh diventa 16, cosi' le linee escono curve e non a zig-zag
SPIGOLI, _TRI_MEDI = _suddividi(TRIANGOLI, 478)
N_PUNTINI = 478 + len(SPIGOLI)                # i puntini usano solo il primo livello
SPIGOLI_2, TRI_FINI = _suddividi(_TRI_MEDI, N_PUNTINI)
N_VERTICI = N_PUNTINI + len(SPIGOLI_2)


def _distanza_dal_bordo():
    """Per ogni vertice, quanti spigoli lo separano dal contorno esterno del volto."""
    vicini = {}
    for a, b in SPIGOLI:
        vicini.setdefault(int(a), set()).add(int(b))
        vicini.setdefault(int(b), set()).add(int(a))
    dist = np.full(478, 99.0, np.float32)
    fronte = ({c.start for c in _FLC.FACE_LANDMARKS_FACE_OVAL}
              | {c.end for c in _FLC.FACE_LANDMARKS_FACE_OVAL})
    passo = 0
    while fronte:
        for v in fronte:
            dist[v] = passo
        fronte = {n for v in fronte for n in vicini.get(v, ()) if dist[n] > passo + 1}
        passo += 1
    dist[dist > 50] = 5.0                       # punti isolati (iridi): mai sfumati
    for spigoli in (SPIGOLI, SPIGOLI_2):        # i punti medi prendono la media degli estremi
        dist = np.concatenate([dist, (dist[spigoli[:, 0]] + dist[spigoli[:, 1]]) / 2])
    return dist


DAL_BORDO = _distanza_dal_bordo()

# tratti evidenziati: segmenti di labbra, occhi e sopracciglia
SEGMENTI = np.array(
    [[c.start, c.end] for gruppo in (
        _FLC.FACE_LANDMARKS_LIPS, _FLC.FACE_LANDMARKS_LEFT_EYE, _FLC.FACE_LANDMARKS_RIGHT_EYE,
        _FLC.FACE_LANDMARKS_LEFT_EYEBROW, _FLC.FACE_LANDMARKS_RIGHT_EYEBROW) for c in gruppo],
    np.int32)
# corna: tronchi di cono curvi. (anelli lungo il corno, punti per anello, punto d'attacco)
CORNA_SEZIONI, CORNA_LATI = 18, 14
CORNA_ATTACCHI = (67, 297)            # punti alti della fronte, uno per lato


def _triangoli_corna():
    tri = []
    per_corno = (CORNA_SEZIONI + 1) * CORNA_LATI
    for c in range(2):
        for i in range(CORNA_SEZIONI):
            for j in range(CORNA_LATI):
                a = c * per_corno + i * CORNA_LATI + j
                b = c * per_corno + i * CORNA_LATI + (j + 1) % CORNA_LATI
                tri += [(a, b, a + CORNA_LATI), (b, b + CORNA_LATI, a + CORNA_LATI)]
    return np.array(tri, np.int32)


TRI_CORNA = _triangoli_corna()
ALI_RIGHE = 42                # righe di caratteri sull'altezza dello schermo (meno = piu' grandi)
GLIFI = " .:;!/|*#%@"         # caratteri delle ali, dal piu' vuoto al piu' pieno


def atlante_glifi(cella=(40, 64)):
    """Una striscia con tutti i caratteri, uno per cella: lo shader delle ali pesca da qui."""
    w, h = cella
    img = np.zeros((h, w * len(GLIFI)), np.uint8)
    for i, c in enumerate(GLIFI):
        (lw, lh), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
        cv2.putText(img, c, (i * w + (w - lw) // 2, (h + lh) // 2), cv2.FONT_HERSHEY_DUPLEX,
                    1.5, 255, 2, cv2.LINE_AA)
    return img
COLONNE_DENTI = 33    # suddivisioni di ogni arcata di denti
N_IMPULSI = 6         # onde della voce visibili contemporaneamente
# cursori del pannello: (etichetta, attributo di Ologramma, minimo, massimo)
CURSORI = [
    ("Linee", "n_linee", 8.0, 60.0),
    ("Spessore", "spessore", 0.6, 3.0),
    ("Bagliore", "bagliore", 0.0, 2.5),
    ("Scia", "scia", 0.0, 0.95),
    ("Prospettiva", "prospettiva", 0.0, 1.6),
    ("Dondolio", "oscillazione", 0.0, 0.30),
    ("Grandezza", "grandezza", 0.30, 0.90),
    ("Tinta", "tinta", 0.0, 1.0),
    ("Saturazione", "saturazione", 0.0, 1.0),
    ("Corna", "corna", 0.0, 1.6),
    ("Ali", "ali", 0.0, 1.5),
    ("Bordo sfumato", "sfumatura", 0.0, 1.0),
    ("Luce webcam", "luce_sfondo", 0.0, 1.0),
]
DA_SALVARE = [c[1] for c in CURSORI] + ["stile", "motivo", "iridi", "denti", "tratti", "glitch"]
rng = np.random.default_rng()


# ================================================================ shader
VS_MESH = """
#version 330
uniform float tremolio, tempo;
in vec3 in_pos;
in float in_campo;
in float in_bocca;
in vec3 in_n;
in float in_bordo;
out float v_bordo;
out float v_campo;
out float v_bocca;
out vec3 v_n;
out vec2 v_ndc;

float caso(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec2 scossa = vec2(caso(in_pos.xy * 91.7 + tempo), caso(in_pos.yx * 37.3 - tempo)) - 0.5;
    gl_Position = vec4(in_pos.xy + scossa * 0.012 * tremolio, in_pos.z, 1.0);
    v_campo = in_campo;
    v_bocca = in_bocca;
    v_n = in_n;
    v_ndc = in_pos.xy;
    v_bordo = in_bordo;
}
"""

FS_MESH = """
#version 330
uniform vec3 colore;
uniform float n_linee, fase, spessore, energia, y_fascio, mostra_linee, tempo;
uniform float rivela, in_formazione, voce, sfumatura;
in float v_bordo;                   // 0 sul contorno del volto, cresce verso l'interno
uniform vec2 impulsi[6];            // (eta' in secondi, ampiezza) delle onde della voce
in float v_campo;
in float v_bocca;
in vec3 v_n;
in vec2 v_ndc;
out vec4 f;

float caso(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

void main() {
    if (v_campo > rivela) discard;                             // parte non ancora materializzata

    float fronte = clamp(-normalize(v_n).z, 0.0, 1.0);        // 1 = rivolto alla camera

    // linee di spessore costante in pixel sul campo scelto (profondita', anelli o scansione)
    float l = n_linee * pow(max(v_campo, 1e-4), 0.75) - fase;
    float pendenza = max(fwidth(l), 1e-5);
    float numero = floor(l + 0.5);                             // quale linea e' questa
    float sp = spessore * (0.65 + 0.7 * fronte);               // piu' spesse verso la camera
    float d = abs(fract(l - 0.5) - 0.5) / pendenza;
    float linea = 1.0 - smoothstep(sp - 0.6, sp + 0.6, d);

    // linee secondarie sottili tra una principale e l'altra, solo dove c'e' spazio
    float d3 = abs(fract(l * 3.0 - 0.5) - 0.5) / (pendenza * 3.0);
    float sottili = (1.0 - smoothstep(0.2, 0.9, d3)) * 0.20 * (1.0 - smoothstep(0.08, 0.20, pendenza));

    // ogni linea respira per conto suo; un'onda di luce corre verso l'esterno
    float respiro = 0.75 + 0.25 * sin(tempo * (1.5 + 3.0 * caso(numero)) + 6.283 * caso(numero + 7.0));
    float onda = exp(-pow((fract(l / 9.0 - tempo * 0.22) - 0.5) * 5.0, 2.0));

    // dove le linee sarebbero piu' fitte dei pixel le sostituisco con una luce uniforme
    float fitte = smoothstep(0.35, 0.60, pendenza);
    float luce = mix(linea * respiro * (1.0 + (0.8 + 0.6 * energia) * onda), 0.35, fitte) + sottili;
    luce *= mostra_linee * (1.0 + 0.25 * voce);

    // voce: anelli di luce che si allargano dalla bocca
    float anelli = 0.0;
    for (int i = 0; i < 6; i++) {
        float raggio = impulsi[i].x * 1.1;
        anelli += impulsi[i].y * exp(-impulsi[i].x * 2.4)
                * exp(-pow((v_bocca - raggio) / 0.04, 2.0));
    }

    float bordo = pow(1.0 - fronte, 3.0) * mostra_linee;      // alone sul profilo
    float fascio = exp(-pow((v_ndc.y - y_fascio) / 0.035, 2.0));

    vec3 c = colore * (luce * (0.30 + 0.95 * fronte) + bordo * 0.30 + 0.012 + 0.03 * fronte * fronte);
    c += vec3(1.0) * linea * onda * fronte * 0.12 * mostra_linee;   // le creste dell'onda sbiancano
    c += colore * fascio * (0.04 + 0.9 * linea * mostra_linee);
    c += (colore * 0.6 + 0.4) * anelli * (0.12 + 0.9 * linea);

    // fronte della materializzazione: una riga bianca che avanza
    float soglia = exp(-pow((rivela - v_campo) / 0.03, 2.0)) * in_formazione;
    c += (colore * 0.4 + 0.6) * soglia * 1.6;

    // il volto svanisce verso il contorno invece di finire con un taglio netto
    c *= mix(1.0, smoothstep(0.0, 1.5, v_bordo), sfumatura);

    f = vec4(c * (1.0 + 0.25 * energia), 1.0);
}
"""

VS_PUNTI = """
#version 330
uniform float scala, rivela;
in vec3 in_pos;
in float in_lum;
in float in_campo;
out float v_lum;
void main() {
    gl_Position = vec4(in_pos.xy, in_pos.z - 0.004, 1.0);
    gl_PointSize = scala * (0.6 + in_lum);
    v_lum = in_campo > rivela ? 0.0 : in_lum;
}
"""

FS_PUNTI = """
#version 330
uniform vec3 colore;
in float v_lum;
out vec4 f;
void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.0, d);
    f = vec4((colore * 0.7 + 0.3) * a * a * v_lum * 0.9, 1.0);
}
"""

VS_TRATTI = """
#version 330
in vec3 in_pos;
in float in_lato;
out float v_lato;
void main() {
    gl_Position = vec4(in_pos.xy, in_pos.z - 0.006, 1.0);
    v_lato = in_lato;
}
"""

FS_TRATTI = """
#version 330
uniform vec3 colore;
uniform float forza;
in float v_lato;
out vec4 f;
void main() {
    float a = pow(max(1.0 - v_lato * v_lato, 0.0), 1.5);
    f = vec4((colore * 0.55 + 0.45) * a * forza, 1.0);
}
"""

VS_DENTI = """
#version 330
in vec3 in_pos;
in vec2 in_uv;
in vec2 in_par;
out vec2 v_uv;
out vec2 v_par;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    v_uv = in_uv;
    v_par = in_par;
}
"""

FS_DENTI = """
#version 330
uniform vec3 colore;
uniform float forza;
in vec2 v_uv;      // x: posizione in "denti" dal centro, y: 1 sulla punta, <0 verso la gengiva
in vec2 v_par;     // x: 0 al centro, 1 in fondo all'arcata; y: larghezza/altezza di un dente
out vec4 f;

float rettangolo(vec2 p, vec2 meta, float r) {
    vec2 q = abs(p) - meta + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    float x = fract(v_uv.x) - 0.5;                             // posizione dentro il singolo dente
    vec2 p = vec2(x * v_par.y, v_uv.y);
    float d = rettangolo(p, vec2(v_par.y * 0.5 - 0.035, 0.96), 0.16);
    float aa = fwidth(d);
    float tratto = max(0.014, aa * 0.8);
    float contorno = 1.0 - smoothstep(tratto, tratto + aa * 1.5, abs(d));
    float dentro = 1.0 - smoothstep(-aa, aa, d);
    float luce = contorno * 0.95 + dentro * (0.10 + 0.25 * smoothstep(0.1, 1.0, v_uv.y));
    luce *= smoothstep(-0.45, -0.05, v_uv.y);                  // sfuma verso la gengiva
    luce *= 1.0 - smoothstep(0.55, 1.0, v_par.x);              // e verso il fondo della bocca
    f = vec4((colore * 0.6 + 0.4) * luce * forza, 1.0);
}
"""

FS_GOLA = """
#version 330
uniform vec3 colore;
uniform float forza, tempo;
in vec2 v_uv;      // coordinate nel piano della bocca, in larghezze di volto
in vec2 v_par;
out vec4 f;
void main() {
    // griglia tenue sul fondo della bocca, che si stringe verso la gola
    vec2 q = v_uv / (0.030 + 0.25 * dot(v_uv, v_uv));
    vec2 g = abs(fract(q - 0.5) - 0.5) / max(fwidth(q), vec2(1e-5));
    float griglia = 1.0 - clamp(min(g.x, g.y) - 0.3, 0.0, 1.0);
    float fondo = exp(-dot(v_uv, v_uv) / 0.02);
    float luce = griglia * (0.05 + 0.10 * fondo) * (0.8 + 0.2 * sin(tempo * 3.0 + q.y));
    f = vec4(colore * luce * forza, 1.0);
}
"""

FS_ALI = """
#version 330
uniform sampler2D atlante;
uniform vec3 colore;
uniform vec2 cella;                 // misura di un carattere, in pixel
uniform float tempo, forza, n_glifi;
uniform sampler2D sagoma;           // dove sta la persona nell'immagine della webcam (1 = persona)
uniform float usa_sagoma, zoom, aspetto;
uniform vec2 centro, scena;         // inquadratura attuale e misura della scena in pixel
in vec2 v_uv;                       // x: 0 all'attacco, 1 in punta; y: 0 in basso, 1 in alto
in vec2 v_par;
out vec4 f;

float caso(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

float segmento(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}

// distanza con segno da un poligono (negativa dentro)
float poligono(vec2 p, vec2 v[6]) {
    float d = dot(p - v[0], p - v[0]);
    float s = 1.0;
    for (int i = 0, j = 5; i < 6; j = i, i++) {
        vec2 e = v[j] - v[i], w = p - v[i];
        vec2 b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
        d = min(d, dot(b, b));
        bvec3 c = bvec3(p.y >= v[i].y, p.y < v[j].y, e.x * w.y > e.y * w.x);
        if (all(c) || all(not(c))) s = -s;
    }
    return s * sqrt(d);
}

// incavo ad arco tra due punte: positivo dentro il cerchio da togliere
float incavo(vec2 p, vec2 a, vec2 b, float fondo) {
    vec2 m = (a + b) * 0.5, e = b - a;
    vec2 n = normalize(vec2(e.y, -e.x));
    float corda = length(e) * 0.5;
    vec2 c = m - n * fondo;                                    // centro oltre il bordo dell'ala
    return sqrt(corda * corda + fondo * fondo) - length(p - c);
}

// quanto e' "piena" l'ala in un punto: 0 fuori, fino a 1 sulle ossa
float ala(vec2 uv) {
    vec2 p = vec2(uv.x * 1.9, (uv.y - 0.41) * 2.2);
    vec2 O = vec2(0.0, 0.0), W = vec2(0.50, 0.92), F0 = vec2(0.50, 1.28);
    vec2 F1 = vec2(1.86, 1.18), F2 = vec2(1.78, 0.38), F3 = vec2(1.30, -0.28), B = vec2(0.14, -0.62);
    vec2 v[6] = vec2[6](O, W, F1, F2, F3, B);
    float d = poligono(p, v);
    // bordo d'uscita a festoni, come un'ala di pipistrello
    d = max(d, incavo(p, F1, F2, 0.55));
    d = max(d, incavo(p, F2, F3, 0.50));
    d = max(d, incavo(p, F3, B, 0.75));
    float ossa = min(min(segmento(p, O, W), segmento(p, W, F1)),
                     min(min(segmento(p, W, F2), segmento(p, W, F3)), segmento(p, W, F0)));
    float osso = exp(-pow(ossa / 0.035, 2.0));
    if (d > 0.0) return osso * 0.95;                           // fuori: resta solo l'artiglio
    float orlo = exp(-pow(d / 0.05, 2.0));
    float membrana = 0.34 + 0.16 * sin(length(p - W) * 9.0 - tempo * 2.5);
    return clamp(max(osso, max(orlo * 0.75, membrana)), 0.0, 1.0);
}

void main() {
    // griglia di caratteri: l'ala si valuta al centro di ogni cella, non pixel per pixel
    vec2 cellaxy = floor(gl_FragCoord.xy / cella);
    vec2 scarto = (cellaxy + 0.5) * cella - gl_FragCoord.xy;
    vec2 uv = v_uv + dFdx(v_uv) * scarto.x + dFdy(v_uv) * scarto.y;
    float pieno = ala(uv);
    // con la webcam sullo sfondo le ali stanno dietro al corpo: dove c'e' la persona, il
    // carattere sparisce (sullo sfondo nero il corpo non si vede, quindi restano intere).
    // Dal centro della cella risalgo al punto corrispondente dell'immagine della webcam.
    vec2 s = (cellaxy + 0.5) * cella / scena;
    vec2 u = vec2(s.x * aspetto, 1.0 - s.y);
    vec2 cam = (u - vec2(aspetto * 0.5, 0.5)) / zoom + centro;
    float persona = texture(sagoma, vec2(cam.x / aspetto, cam.y)).r;
    pieno *= 1.0 - usa_sagoma * smoothstep(0.35, 0.60, persona);
    if (pieno <= 0.02) discard;
    // i caratteri sfarfallano: ogni cella cambia un po' qualche volta al secondo
    pieno = clamp(pieno * (0.75 + 0.5 * caso(cellaxy + floor(tempo * 7.0))), 0.0, 0.999);
    float glifo = floor(pieno * n_glifi);
    vec2 q = fract(gl_FragCoord.xy / cella);
    float a = texture(atlante, vec2((glifo + q.x) / n_glifi, 1.0 - q.y)).r;
    f = vec4((colore * 0.85 + 0.15) * a * (0.25 + 0.75 * pieno) * forza, 1.0);
}
"""

VS_IRIDE = """
#version 330
in vec3 in_pos;
in float in_dim;
in float in_aperto;
out float v_aperto;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    gl_PointSize = in_dim;
    v_aperto = in_aperto;
}
"""

FS_IRIDE = """
#version 330
uniform vec3 colore;
uniform float tempo;
in float v_aperto;
out vec4 f;
void main() {
    vec2 q = (gl_PointCoord - 0.5) * 2.0;
    float d = length(q) * 1.5;                                 // 1 = bordo dell'iride
    float anello = exp(-pow((d - 0.92) / 0.10, 2.0));
    float interno = exp(-pow((d - 0.52) / 0.05, 2.0)) * 0.5;
    float pupilla = exp(-pow(d / 0.20, 2.0)) * 1.3;
    float raggi = pow(0.5 + 0.5 * cos(atan(q.y, q.x) * 16.0 + tempo * 0.8), 3.0)
                * smoothstep(0.50, 0.62, d) * (1.0 - smoothstep(0.80, 0.92, d)) * 0.35;
    float luce = (anello + interno + pupilla + raggi) * (1.0 - smoothstep(1.2, 1.45, d));
    f = vec4((colore * 0.8 + 0.2) * luce * v_aperto, 1.0);
}
"""

VS_SCHERMO = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_pos * 0.5 + 0.5;
}
"""

FS_SCIA = """
#version 330
uniform sampler2D nuovo, vecchio;
uniform float tenuta;
in vec2 uv;
out vec4 f;
void main() {
    f = max(texture(nuovo, uv), texture(vecchio, uv) * tenuta);
}
"""

FS_SFOCA = """
#version 330
uniform sampler2D tex;
uniform vec2 passo;
in vec2 uv;
out vec4 f;
void main() {
    vec4 c = texture(tex, uv) * 0.227027;
    c += (texture(tex, uv + passo * 1.3846) + texture(tex, uv - passo * 1.3846)) * 0.316216;
    c += (texture(tex, uv + passo * 3.2308) + texture(tex, uv - passo * 3.2308)) * 0.070270;
    f = c;
}
"""

VS_PANNELLO = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_uv;
}
"""

FS_PANNELLO = """
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 f;
void main() {
    f = texture(tex, uv);
}
"""

FS_FINALE = """
#version 330
uniform sampler2D scena, a0, a1, a2, a3, a4, a5, webcam;
uniform float t, seme, glitch, energia, sfondo, bagliore, lampo, colpo, rapporto, luce_sfondo;
uniform vec2 dimensione;
in vec2 uv;
out vec4 f;

float caso(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec3 luce(vec2 p) {
    vec3 alone = 0.30 * texture(a0, p).rgb + 0.30 * texture(a1, p).rgb + 0.28 * texture(a2, p).rgb
               + 0.26 * texture(a3, p).rgb + 0.24 * texture(a4, p).rgb + 0.22 * texture(a5, p).rgb;
    return texture(scena, p).rgb + bagliore * alone;
}

void main() {
    vec2 p = uv;
    // glitch: alcune bande orizzontali scivolano di lato (sempre, sull'occhiolino)
    float banda = floor(p.y * 28.0 + seme * 7.0);
    if (caso(vec2(banda, seme)) < max(glitch, colpo * 0.7))
        p.x += (caso(vec2(seme, banda)) - 0.5) * (0.14 + 0.2 * colpo);

    // aberrazione cromatica, piu' forte verso i bordi e quando apri la bocca
    float ab = (0.0010 + 0.0015 * energia + 0.008 * colpo) * rapporto * (0.5 + length(uv - 0.5) * 2.0);
    vec3 c = vec3(luce(p + vec2(ab, 0.0)).r, luce(p).g, luce(p - vec2(ab, 0.0)).b);

    c *= 1.0 + 0.8 * lampo;                                    // sorpresa: lampo...
    c = mix(c, vec3(dot(c, vec3(0.45))), lampo * 0.7);          // ...che sbianca tutto

    c = 1.0 - exp(-c * 1.5);                                   // compressione dei toni
    c *= 0.86 + 0.14 * sin(uv.y * dimensione.y * 3.14159);     // righe di scansione
    c *= 1.0 - 0.45 * pow(length(uv - 0.5) * 1.3, 2.5);        // vignettatura
    c += (caso(uv * dimensione + t) - 0.5) * 0.035;            // grana
    c *= 0.94 + 0.06 * caso(vec2(t, 3.0));                     // sfarfallio

    vec3 fondo = texture(webcam, vec2(uv.x, 1.0 - uv.y)).bgr * luce_sfondo * sfondo;
    f = vec4(fondo + c, 1.0);
}
"""


# ================================================================ rendering
def imposta(programma, **valori):
    """Assegna gli uniform ignorando quelli che lo shader non usa."""
    for nome, valore in valori.items():
        if nome in programma:
            programma[nome].value = valore


def vertici(ctx, programma, buffer, formato, *attributi, indici=None):
    """Crea un vertex array saltando gli attributi che il driver ha eliminato dallo shader
    perche' inutilizzati (al loro posto mette un riempimento della stessa dimensione)."""
    pezzi, nomi = [], []
    for pezzo, nome in zip(formato.split(), attributi):
        if nome in programma:
            pezzi.append(pezzo)
            nomi.append(nome)
        else:
            pezzi.append(f"{int(pezzo[:-1]) * 4}x")
    return ctx.vertex_array(programma, [(buffer, " ".join(pezzi), *nomi)], indici)


class Ologramma:
    """Stato dell'effetto (posizioni, espressioni, voce) e tutte le passate di rendering."""

    def __init__(self, ctx, larghezza, altezza):
        self.ctx = ctx
        self.dim = (larghezza, altezza)
        self.ss = max(1, int(SOVRACAMPIONA))         # fattore di sovracampionamento della scena
        self.aspetto = larghezza / altezza
        self.centro_schermo = np.array([self.aspetto / 2, 0.5, 0.0], np.float32)

        # ---- opzioni (cambiate dai tasti)
        self.stile, self.motivo = "ologramma", "livello"
        self.glitch, self.sfondo = True, False
        self.iridi = self.denti = self.tratti = True
        self.adatta = True                # volto centrato, ingrandito e in prospettiva
        self.grandezza, self.prospettiva = GRANDEZZA, PROSPETTIVA
        self.oscillazione, self.scia = OSCILLAZIONE, SCIA
        self.sfumatura = 1.0              # 0 = bordo netto, 1 = il volto svanisce verso il contorno
        self.corna = 1.0                  # lunghezza delle corna (0 = nessuna)
        self._corna_prima = 1.0
        self.ali = 1.0                    # grandezza delle ali (0 = nessuna)
        self._ali_prima = 1.0
        self.spalle = None                # (2, 2) spalle tracciate, in coordinate normalizzate
        self._spalle_viste = -99.0
        self.ha_sagoma = False
        self.luce_sfondo = 0.30           # luminosita' della webcam quando fa da sfondo
        self.rapporto = 1.0               # grandezza del volto a schermo rispetto a quella "piena"
        self.preset(1)

        # ---- stato
        self.pos = None                   # (478, 3) coordinate normalizzate 0..1 (z come x)
        self.vel = None
        self.presenza = 0.0               # 0 = volto assente, 1 = completamente materializzato
        self.energia = self.sopracciglia = self.rabbia = 0.0
        self.lampo = self.colpo = 0.0     # lampo di sorpresa, glitch da occhiolino
        self._sorpreso = self._occhiolino = False
        self.fase = 0.0
        self.occhi = np.ones(2, np.float32)          # apertura di ciascun occhio (0 = chiuso)
        self.centro = self.centro_schermo.copy()
        self.zoom, self.spazio = 1.0, 0.0            # spazio: 0 = piatto, 1 = prospettiva piena
        # voce
        self.voce = self._voce_lenta = 0.0
        self._picco = 0.02
        self.impulsi = np.zeros((N_IMPULSI, 2), np.float32)       # (eta', ampiezza)
        self.impulsi[:, 0] = 99.0
        self._ultimo_impulso = 0.0
        # denti: non sono tracciati, li ricostruisco. Le due misure (in larghezze di volto)
        # si tarano da sole ogni volta che la bocca e' chiusa.
        self.dente_su = 0.125             # dalla base del naso alla linea delle labbra
        self.dente_giu = 0.320            # dalla linea delle labbra al mento

        # ---- programmi
        P = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_mesh = P(VS_MESH, FS_MESH)
        self.p_punti = P(VS_PUNTI, FS_PUNTI)
        self.p_tratti = P(VS_TRATTI, FS_TRATTI)
        self.p_denti = P(VS_DENTI, FS_DENTI)
        self.p_gola = P(VS_DENTI, FS_GOLA)
        self.p_ali = P(VS_DENTI, FS_ALI)
        self.p_iride = P(VS_IRIDE, FS_IRIDE)
        self.p_scia = P(VS_SCHERMO, FS_SCIA)
        self.p_sfoca = P(VS_SCHERMO, FS_SFOCA)
        self.p_finale = P(VS_SCHERMO, FS_FINALE)

        # ---- geometria
        self.b_mesh = ctx.buffer(reserve=N_VERTICI * 9 * 4, dynamic=True)
        self.b_punti = ctx.buffer(reserve=N_PUNTINI * 5 * 4, dynamic=True)
        self.b_tratti = ctx.buffer(reserve=len(SEGMENTI) * 6 * 4 * 4, dynamic=True)
        self.b_denti = ctx.buffer(reserve=4 * COLONNE_DENTI * 7 * 4, dynamic=True)
        self.b_gola = ctx.buffer(reserve=6 * 7 * 4, dynamic=True)
        self.b_corna = ctx.buffer(reserve=2 * (CORNA_SEZIONI + 1) * CORNA_LATI * 9 * 4, dynamic=True)
        self.b_iridi = ctx.buffer(reserve=2 * 5 * 4, dynamic=True)
        indici = ctx.buffer(TRI_FINI.astype("i4").tobytes())
        A = lambda prog, buf, formato, *nomi, indici=None: vertici(
            ctx, prog, buf, formato, *nomi, indici=indici)
        self.v_corna = A(self.p_mesh, self.b_corna, "3f 1f 1f 3f 1f",
                         "in_pos", "in_campo", "in_bocca", "in_n", "in_bordo",
                         indici=ctx.buffer(TRI_CORNA.astype("i4").tobytes()))
        self.v_mesh = A(self.p_mesh, self.b_mesh, "3f 1f 1f 3f 1f",
                        "in_pos", "in_campo", "in_bocca", "in_n", "in_bordo", indici=indici)
        self.v_punti = A(self.p_punti, self.b_punti, "3f 1f 1f", "in_pos", "in_lum", "in_campo")
        self.v_tratti = A(self.p_tratti, self.b_tratti, "3f 1f", "in_pos", "in_lato")
        k = np.arange(COLONNE_DENTI - 1)
        quadri = np.stack([2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2], 1)
        quadri = np.vstack([quadri, quadri + 2 * COLONNE_DENTI])        # due arcate
        self.v_denti = A(self.p_denti, self.b_denti, "3f 2f 2f", "in_pos", "in_uv", "in_par",
                         indici=ctx.buffer(quadri.astype("i4").tobytes()))
        self.v_gola = A(self.p_gola, self.b_gola, "3f 2f 2f", "in_pos", "in_uv", "in_par")
        self.b_ali = ctx.buffer(reserve=12 * 7 * 4, dynamic=True)
        self.v_ali = A(self.p_ali, self.b_ali, "3f 2f 2f", "in_pos", "in_uv", "in_par")
        glifi = atlante_glifi()
        self.t_glifi = ctx.texture(glifi.shape[::-1], 1, glifi.tobytes(), dtype="f1")
        self.t_glifi.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_glifi.repeat_x = self.t_glifi.repeat_y = False
        self.t_sagoma = ctx.texture((256, 144), 1, dtype="f1")
        self.t_sagoma.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_sagoma.repeat_x = self.t_sagoma.repeat_y = False
        self.v_iridi = A(self.p_iride, self.b_iridi, "3f 1f 1f", "in_pos", "in_dim", "in_aperto")
        quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        V = lambda prog: A(prog, quad, "2f", "in_pos")
        self.q_scia, self.q_sfoca, self.q_finale = V(self.p_scia), V(self.p_sfoca), V(self.p_finale)

        # ---- framebuffer (HDR a 16 bit)
        def bersaglio(dim, profondita=False):
            tex = ctx.texture(dim, 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            dep = ctx.depth_renderbuffer(dim) if profondita else None
            return ctx.framebuffer([tex], dep), tex

        # la scena e' piu' grande dello schermo: ridurla dopo ammorbidisce bordi e puntini
        self.f_scena, self.t_scena = bersaglio((larghezza * self.ss, altezza * self.ss), True)
        self.scie = [bersaglio(self.dim), bersaglio(self.dim)]
        # piramide del bagliore: ogni livello e' la meta' del precedente
        self.aloni = []
        for i in range(1, N_ALONI + 1):
            dim = (max(2, larghezza >> i), max(2, altezza >> i))
            self.aloni.append([bersaglio(dim), bersaglio(dim)])
        self.f_uscita = ctx.framebuffer([ctx.texture(self.dim, 3)])   # per video e screenshot
        self.t_webcam = None

    # ------------------------------------------------------------ opzioni
    def preset(self, numero):
        nome, self.tinta, self.saturazione, self.n_linee, self.bagliore, self.spessore = PRESET[numero]
        return nome

    def carica_sagoma(self, maschera):
        """maschera: uint8 (144, 256), 255 dove c'e' la persona. None = non disponibile."""
        self.ha_sagoma = maschera is not None
        if self.ha_sagoma:
            self.t_sagoma.write(np.ascontiguousarray(maschera).tobytes())

    def carica_webcam(self, fotogramma):
        h, w = fotogramma.shape[:2]
        if self.t_webcam is None or self.t_webcam.size != (w, h):
            self.t_webcam = self.ctx.texture((w, h), 3, dtype="f1")
        self.t_webcam.write(np.ascontiguousarray(fotogramma).tobytes())

    def leggi_immagine(self):
        """L'immagine finale come array BGR (per screenshot, video e webcam virtuale)."""
        w, h = self.dim
        self.componi(self.f_uscita, (0, 0, w, h))
        dati = np.frombuffer(self.f_uscita.read(components=3), np.uint8).reshape(h, w, 3)
        return dati[::-1, :, ::-1]

    # ------------------------------------------------------------ fisica
    def aggiorna(self, punti, espressioni, dt, livello_audio=0.0, t=0.0, spalle=None):
        if spalle is not None:             # spalle tracciate: le inseguo con dolcezza
            if self.spalle is None or t - self._spalle_viste > 1.0:
                self.spalle = spalle.copy()
            self.spalle += min(1.0, dt * 12.0) * (spalle - self.spalle)
            self._spalle_viste = t
        self._t = t
        e = espressioni or {}
        media = lambda a, b: (e.get(a, 0.0) + e.get(b, 0.0)) / 2
        k = min(1.0, dt * 9.0)
        self.energia += k * (e.get("jawOpen", 0.0) - self.energia)
        self.sopracciglia += k * (e.get("browInnerUp", 0.0) - self.sopracciglia)
        rabbia = float(np.clip((media("browDownLeft", "browDownRight") - 0.25) / 0.45, 0, 1))
        self.rabbia += min(1.0, dt * 5.0) * (rabbia - self.rabbia)
        chiusi = [e.get("eyeBlinkRight", 0.0), e.get("eyeBlinkLeft", 0.0)]
        for i in range(2):
            aperto = 1.0 - float(np.clip(chiusi[i] * 1.6, 0, 1))
            self.occhi[i] += min(1.0, dt * 25.0) * (aperto - self.occhi[i])
        self.fase += dt * (0.35 + 1.5 * self.sopracciglia + 0.5 * self.energia + 0.4 * self.voce)

        # sorpresa -> lampo; occhiolino -> glitch secco (scattano una volta, poi si spengono)
        sorpreso = media("eyeWideLeft", "eyeWideRight") > 0.45
        if sorpreso and not self._sorpreso:
            self.lampo = 1.0
        occhiolino = max(chiusi) > 0.6 and min(chiusi) < 0.3
        if occhiolino and not self._occhiolino:
            self.colpo = 1.0
        self._sorpreso, self._occhiolino = sorpreso, occhiolino
        self.lampo = max(0.0, self.lampo - dt * 2.5)
        self.colpo = max(0.0, self.colpo - dt * 5.0)

        self._aggiorna_voce(livello_audio, dt, t)

        if punti is not None:
            if self.pos is None or self.presenza <= 0.0:
                # volto nuovo: parte gia' al posto giusto, niente volo attraverso lo schermo
                self.pos, self.vel = punti.copy(), np.zeros_like(punti)
                self._inquadra(1.0)
            # molla smorzata: la maschera insegue il volto con un filo di inerzia
            self.vel += (400.0 * (punti - self.pos) - 34.0 * self.vel) * dt
            self.pos += self.vel * dt
            self.presenza = min(1.0, self.presenza + dt * 1.1)

            # taratura dei denti: a bocca chiusa misuro dove si incontrano le labbra
            U = self._unita(self.pos)
            _, giu, _, L = self._terna(U)
            if (U[14] - U[13]) @ giu / L < 0.02 and self.energia < 0.08:
                kk = min(1.0, dt * 2.0)
                self.dente_su += kk * ((U[13] - U[2]) @ giu / L - self.dente_su)
                self.dente_giu += kk * ((U[152] - U[14]) @ giu / L - self.dente_giu)
            self._inquadra(min(1.0, dt * 6.0))
        elif self.pos is not None:
            # volto perso: resta fermo e si dissolve da fuori verso il naso
            self.presenza = max(0.0, self.presenza - dt * 1.4)

    def busto(self):
        """Le due spalle (sinistra e destra dell'immagine) in unita' di schermo.
        Se il corpo non e' tracciato le stimo dal volto: sotto il mento, larghe poco piu' di due teste."""
        if self.spalle is not None and getattr(self, "_t", 0.0) - self._spalle_viste < 1.0:
            return self._unita(np.hstack([self.spalle, np.zeros((2, 1), np.float32)]))
        alto, basso = self._unita(self.pos[10]), self._unita(self.pos[152])
        testa = np.linalg.norm(basso - alto)
        centro = (alto + basso) / 2 + np.float32([0, 1.0, 0]) * testa
        centro[2] = 0.0
        lato = np.float32([1.1, 0, 0]) * testa
        return np.stack([centro - lato, centro + lato])

    def _inquadra(self, k):
        """Centro e zoom inseguono il volto (o tornano alla posizione reale)."""
        if self.adatta:
            alto, basso = self._unita(self.pos[10]), self._unita(self.pos[152])
            # con corna e ali l'inquadratura si allarga e sale, per non tagliarle
            spazio_corna = min(self.corna, 1.6) * 0.26
            centro = (alto + basso) / 2 - (basso - alto) * (spazio_corna * 0.5)
            zoom = self.grandezza / (np.linalg.norm(alto - basso) * (1.0 + spazio_corna * 0.6) + 1e-6)
            if self.ali > 0.01:
                # con le ali l'inquadratura comprende anche le spalle, dove sono attaccate
                spalle = self.busto()
                cima = alto + (alto - basso) * (0.30 + spazio_corna)
                fondo = spalle.mean(axis=0) + np.float32([0, 0.32, 0]) * np.linalg.norm(spalle[1] - spalle[0])
                centro = np.float32([centro[0], (cima[1] + fondo[1]) / 2, centro[2]])
                zoom = min(zoom, 0.92 / (abs(fondo[1] - cima[1]) + 1e-6))
        else:
            centro, zoom = self.centro_schermo, 1.0
        self.centro += k * (centro - self.centro)
        self.zoom += k * (zoom - self.zoom)
        self.spazio += k * ((1.0 if self.adatta else 0.0) - self.spazio)

    def _aggiorna_voce(self, livello, dt, t):
        """Dal volume del microfono a un valore 0..1 e agli impulsi sulle sillabe."""
        self._picco = max(self._picco * (1.0 - 0.08 * dt), livello, 0.02)   # guadagno automatico
        v = float(np.clip((livello / self._picco - 0.15) / 0.85, 0.0, 1.0))
        self.voce += min(1.0, dt * (30.0 if v > self.voce else 7.0)) * (v - self.voce)
        self._voce_lenta += min(1.0, dt * 4.0) * (v - self._voce_lenta)
        self.impulsi[:, 0] += dt
        if v - self._voce_lenta > 0.22 and t - self._ultimo_impulso > 0.11:
            self._ultimo_impulso = t
            self.impulsi[np.argmax(self.impulsi[:, 0])] = (0.0, 0.4 + 0.6 * v)   # rimpiazza la piu' vecchia

    # ------------------------------------------------------------ geometria
    @staticmethod
    def _terna(P):
        """Assi solidali con la testa: destra, giu', avanti (verso la camera) e larghezza."""
        destra = P[454] - P[234]
        larghezza = np.linalg.norm(destra) + 1e-6
        destra = destra / larghezza
        giu = P[152] - P[10]
        giu = giu - (giu @ destra) * destra
        giu /= np.linalg.norm(giu) + 1e-9
        avanti = np.cross(destra, giu)
        if avanti[2] > 0:                       # "avanti" deve puntare verso la camera
            avanti = -avanti
        return destra, giu, avanti, larghezza

    def _unita(self, p):
        """Da coordinate normalizzate a uno spazio con unita' uguali sui tre assi."""
        return p * np.array([self.aspetto, 1.0, self.aspetto], np.float32)

    def _proietta(self, Q, t):
        """Camera virtuale: leggero dondolio e prospettiva, poi coordinate dello schermo."""
        q = np.asarray(Q, np.float32) - self.centro_schermo
        if self.spazio > 1e-3:
            imb = self.oscillazione * self.spazio * np.sin(t * 0.50)
            bec = self.oscillazione * 0.5 * self.spazio * np.sin(t * 0.37 + 1.0)
            c, s = np.cos(imb), np.sin(imb)
            q = np.stack([q[:, 0] * c + q[:, 2] * s, q[:, 1], -q[:, 0] * s + q[:, 2] * c], 1)
            c, s = np.cos(bec), np.sin(bec)
            q = np.stack([q[:, 0], q[:, 1] * c - q[:, 2] * s, q[:, 1] * s + q[:, 2] * c], 1)
            q[:, :2] /= (1.0 + self.prospettiva * self.spazio * q[:, 2:3]).clip(0.3)
        return np.stack([q[:, 0] / self.aspetto * 2, -q[:, 1] * 2,
                         np.clip(q[:, 2] / self.aspetto * 2, -0.98, 0.98)], axis=1)

    def _prepara(self, t):
        """Costruisce tutta la geometria del fotogramma e la carica nei buffer."""
        P = (self._unita(self.pos) - self.centro) * self.zoom + self.centro_schermo
        P0 = P
        # terna solidale con la testa: le linee restano incollate al volto quando ruoti
        destra, giu, avanti, L = self._terna(P)
        dietro = -avanti
        bocca = (P[13] + P[14]) / 2

        # la profondita' stimata e' il dato piu' rumoroso: la liscio un po' tra punti vicini,
        # altrimenti le linee di livello tremano e si spezzano
        for _ in range(2):
            prof = P @ dietro
            somma, quanti = np.zeros(478, np.float32), np.zeros(478, np.float32)
            for a_, b_ in ((SPIGOLI[:, 0], SPIGOLI[:, 1]), (SPIGOLI[:, 1], SPIGOLI[:, 0])):
                np.add.at(somma, a_, prof[b_])
                np.add.at(quanti, a_, 1.0)
            liscia = np.where(quanti > 0, somma / np.maximum(quanti, 1), prof)
            P = P + dietro * ((liscia - prof) * 0.6)[:, None]
        P0 = P

        # normali per vertice, orientate verso l'esterno della testa
        a, b, c = P[TRIANGOLI[:, 0]], P[TRIANGOLI[:, 1]], P[TRIANGOLI[:, 2]]
        nf = np.cross(b - a, c - a)
        centro_testa = P.mean(axis=0) + dietro * L * 0.6
        nf[np.einsum("ij,ij->i", nf, (a + b + c) / 3 - centro_testa) < 0] *= -1
        nv = np.zeros_like(P)
        for k in range(3):
            np.add.at(nv, TRIANGOLI[:, k], nf)
        nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-9

        # suddivisione con superficie arrotondata (Phong tessellation): ogni punto medio
        # viene spinto verso i piani tangenti dei due estremi, cosi' le linee non fanno spigoli
        for spigoli in (SPIGOLI, SPIGOLI_2):
            ia, ib = spigoli[:, 0], spigoli[:, 1]
            medio = (P[ia] + P[ib]) / 2
            su_a = medio - np.einsum("ij,ij->i", medio - P[ia], nv[ia])[:, None] * nv[ia]
            su_b = medio - np.einsum("ij,ij->i", medio - P[ib], nv[ib])[:, None] * nv[ib]
            medio += 0.75 * ((su_a + su_b) / 2 - medio)
            n_medio = nv[ia] + nv[ib]
            n_medio /= np.linalg.norm(n_medio, axis=1, keepdims=True) + 1e-9
            P, nv = np.vstack([P, medio]), np.vstack([nv, n_medio])

        # campo su cui corrono le linee
        if self.motivo == "anelli":            # anelli attorno a occhi, bocca e centro della fronte
            centri = np.stack([P0[468], P0[473], bocca, P0[151]])
            dist = np.linalg.norm(P[:, None, :] - centri[None], axis=2) / L
            campo = -np.log(np.exp(-14.0 * dist).sum(axis=1)) / 14.0 + 0.10
        elif self.motivo == "scansione":       # righe orizzontali che seguono la testa
            campo = ((P - P0[10]) @ giu / L + 0.45) * 0.55
        else:                                  # curve di livello della profondita' (0 sul naso)
            campo = (P - P0[1]) @ dietro / L
        campo = np.clip(campo, 0.0, None)
        dist_bocca = np.linalg.norm(P - bocca, axis=1) / L

        ndc = self._proietta(P, t)
        self.b_mesh.write(np.hstack(
            [ndc, campo[:, None], dist_bocca[:, None], nv, DAL_BORDO[:, None]]
        ).astype("f4").tobytes())

        # puntini: uno per ogni vertice
        z = ndc[:, 2]
        vicinanza = 1.0 - (z - z.min()) / (np.ptp(z) + 1e-6)
        lum = 0.25 + 0.75 * vicinanza
        lum = lum * (1.0 - self.sfumatura * (1.0 - np.clip(DAL_BORDO / 1.5, 0, 1) ** 2))
        self.b_punti.write(np.hstack(
            [ndc, lum[:, None], campo[:, None]])[:N_PUNTINI].astype("f4").tobytes())

        # tratti: ogni segmento diventa una striscia larga qualche pixel
        w, h = self.dim[0] * self.ss, self.dim[1] * self.ss        # pixel della scena
        A, B = ndc[SEGMENTI[:, 0]], ndc[SEGMENTI[:, 1]]
        d = (B[:, :2] - A[:, :2]) * (w / 2, h / 2)
        perp = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        mezza = max(1.5 * self.ss, 0.006 * L * h)                   # mezza larghezza in pixel
        scarto = np.hstack([perp * mezza / (w / 2, h / 2), np.zeros((len(A), 1))])
        vert = lambda Q, lato: np.hstack([Q + scarto * lato, np.full((len(Q), 1), lato)])
        strisce = np.stack([vert(A, -1), vert(A, 1), vert(B, -1),
                            vert(B, -1), vert(A, 1), vert(B, 1)], 1)
        self.b_tratti.write(strisce.astype("f4").tobytes())

        # iridi: un disco luminoso per occhio, appena dietro le palpebre
        pieno = float(np.clip((self.presenza - 0.85) / 0.15, 0, 1))
        dati = np.zeros((2, 5), np.float32)
        dati[:, :3] = self._proietta(P0[[468, 473]], t) + (0.0, 0.0, 0.012)
        raggi = [np.linalg.norm(P0[469:473, :2] - P0[468, :2], axis=1).mean(),
                 np.linalg.norm(P0[474:478, :2] - P0[473, :2], axis=1).mean()]
        dati[:, 3] = np.array(raggi) * h * 2 * 1.5                  # diametro in pixel (+margine)
        dati[:, 4] = self.occhi * pieno
        self.b_iridi.write(dati.tobytes())

        self._prepara_bocca(P0, destra, giu, dietro, L, bocca, t)
        if self.ali > 0.01:
            self._prepara_ali(t)
        if self.corna > 0.01:
            campo_corna = self._prepara_corna(P0, nv, campo, destra, giu, avanti, L, bocca, t)
            return N_PUNTINI, L, max(float(campo.max()), campo_corna), pieno
        return N_PUNTINI, L, float(campo.max()), pieno

    def _prepara_ali(self, t):
        """Due grandi rettangoli attaccati alla schiena, tra le spalle: la forma dell'ala la
        disegna lo shader."""
        # spalle nello stesso spazio (centrato e ingrandito) in cui e' disegnato il volto
        spalle = (self.busto() - self.centro) * self.zoom + self.centro_schermo
        larg = np.linalg.norm(spalle[1, :2] - spalle[0, :2]) + 1e-6
        destra = np.float32([*(spalle[1, :2] - spalle[0, :2]) / larg, 0.0])   # lungo le spalle
        su = np.float32([destra[1], -destra[0], 0.0])                          # verso la testa
        dietro = np.float32([0.0, 0.0, 1.0])
        # si aprono con rabbia, bocca aperta e voce; altrimenti battono piano, un po' raccolte
        apertura = max(self.rabbia, 0.6 * self.energia, 0.5 * self.voce)
        angolo = 0.38 + 0.16 * np.sin(t * 1.4) - 0.34 * apertura
        attacco = spalle.mean(axis=0) - su * (0.15 * larg) + dietro * (0.25 * larg)   # scapole
        misura = self.ali * larg * 0.60
        vertici = []
        for lato in (-1.0, 1.0):
            fuori = (destra * lato) * np.cos(angolo) + dietro * np.sin(angolo)
            origine = attacco + destra * (lato * 0.16 * larg)
            for u, v in ((0, 0), (1, 0), (0, 1), (0, 1), (1, 0), (1, 1)):
                Q = origine + fuori * (u * 1.9 * misura) + su * ((v - 0.41) * 2.2 * misura)
                vertici.append(np.hstack([Q, [u, v, 0.0, 0.0]]))
        dati = np.array(vertici, np.float32)
        dati[:, :3] = self._proietta(dati[:, :3], t)
        dati[:, 2] = 0.9999               # in fondo a tutto: volto, corna e bocca le coprono
        self.b_ali.write(dati.tobytes())

    def _prepara_corna(self, P, nv, campo, destra, giu, avanti, L, bocca, t):
        """Due corna da maschera oni: partono dalla fronte, vanno in avanti e curvano in alto."""
        su = -giu
        s = np.linspace(0.0, 1.0, CORNA_SEZIONI + 1, dtype=np.float32)[:, None]
        angolo = np.linspace(0.0, 2 * np.pi, CORNA_LATI, endpoint=False, dtype=np.float32)
        lung = self.corna * L * 0.62            # a 1.0 sono corte e tozze, da oni
        vertici, campo_max = [], 0.0
        for attacco in CORNA_ATTACCHI:
            lato = np.sign((P[attacco] - P[10]) @ destra) or 1.0        # verso l'esterno del volto
            fuori = destra * lato
            base = P[attacco] - nv[attacco] * (0.03 * L)                # un po' dentro la fronte
            # curva di Bezier: esce in avanti, poi sale e si apre appena
            p1 = base + (avanti * 0.26 + su * 0.16 + fuori * 0.06) * lung
            p2 = base + (avanti * 0.14 + su * 0.56 + fuori * 0.16) * lung
            asse = (1 - s) ** 2 * base + 2 * s * (1 - s) * p1 + s ** 2 * p2
            tang = 2 * (1 - s) * (p1 - base) + 2 * s * (p2 - p1)
            tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
            n1 = np.cross(tang, destra)
            n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-9
            n2 = np.cross(tang, n1)
            raggio = 0.095 * L * (0.35 + 0.65 * min(self.corna, 1.0)) * (1.0 - s) ** 0.8
            # ogni anello: un cerchio attorno all'asse, sempre piu' stretto verso la punta
            giro = (np.cos(angolo)[None, :, None] * n1[:, None, :]
                    + np.sin(angolo)[None, :, None] * n2[:, None, :])    # (sezioni, lati, 3)
            punti = asse[:, None, :] + giro * raggio[:, None, :]
            # le linee proseguono dalla fronte e diventano anelli attorno al corno
            c = campo[attacco] + s * 0.30 * self.corna
            campo_max = max(campo_max, float(c.max()))
            Q = punti.reshape(-1, 3)
            dati = np.hstack([
                self._proietta(Q, t),
                np.repeat(c, CORNA_LATI, axis=0),
                (np.linalg.norm(Q - bocca, axis=1) / L)[:, None],
                giro.reshape(-1, 3),
                np.full((len(Q), 1), 5.0, np.float32)])                 # mai sfumate
            vertici.append(dati)
        self.b_corna.write(np.vstack(vertici).astype("f4").tobytes())
        return campo_max

    def _prepara_bocca(self, P, destra, giu, dietro, L, bocca, t):
        """Denti e fondo della bocca: si vedono solo attraverso l'apertura delle labbra."""
        # ai lati non oltre gli angoli della bocca
        angoli = sorted(((P[78] - P[13]) @ destra / L, (P[308] - P[13]) @ destra / L))

        def arcata(centro, punta, verso, altezza, larg_dente, prof):
            # punta: quota del bordo dei denti; verso: +1 se la gengiva sta sopra, -1 sotto
            scarto = (P[13] - centro) @ destra / L
            x = np.linspace(max(-0.27, angoli[0] + scarto), min(0.27, angoli[1] + scarto),
                            COLONNE_DENTI, dtype=np.float32)[:, None]
            arco = (x / 0.27) ** 2 * 0.10 * L * dietro            # l'arcata curva verso la gola
            bordo = centro + destra * (x * L) + giu * (punta * L) + dietro * prof + arco
            gengiva = bordo - giu * (verso * altezza * 1.45 * L)
            par = np.full((COLONNE_DENTI, 1), larg_dente / altezza, np.float32)
            riga = lambda Q, v: np.hstack([self._proietta(Q, t), x / larg_dente,
                                           np.full_like(x, v), np.abs(x) / 0.27, par])
            return np.stack([riga(bordo, 1.0), riga(gengiva, -0.45)], 1).reshape(-1, 7)

        # superiori: fissi rispetto al cranio (ancorati alla base del naso)
        sopra = arcata(P[2], self.dente_su + 0.012, +1, 0.075, 0.054,
                       (P[13] - P[2]) @ dietro + 0.02 * L)
        # inferiori: seguono la mandibola (ancorati al mento)
        sotto = arcata(P[152], -self.dente_giu, -1, 0.060, 0.042,
                       (P[14] - P[152]) @ dietro + 0.03 * L)
        denti = np.vstack([sopra, sotto]).astype(np.float32)
        # profondita' quasi al fondo: passano il test solo dove la mesh del volto ha un buco
        denti[:, 2] = 0.9990
        self.b_denti.write(denti.tobytes())

        # fondo della bocca: un rettangolo ancora piu' indietro dei denti
        uv = np.float32([[-1, -1], [1, -1], [-1, 1], [-1, 1], [1, -1], [1, 1]]) * (0.32, 0.30)
        Q = bocca + destra * (uv[:, :1] * L) + giu * (uv[:, 1:] * L) + dietro * (0.12 * L)
        gola = np.hstack([self._proietta(Q, t), uv, np.zeros((6, 2), np.float32)]).astype(np.float32)
        gola[:, 2] = 0.9995
        self.b_gola.write(gola.tobytes())

    # ------------------------------------------------------------ disegno
    def disegna(self, t, dt):
        ctx, (w, h) = self.ctx, self.dim
        colore = np.array(colorsys.hsv_to_rgb(self.tinta, self.saturazione, 1.0))
        colore += self.rabbia * 0.9 * (np.array([1.0, 0.10, 0.06]) - colore)   # rabbia: rosso
        colore = tuple(float(v) for v in colore)
        tempo = t % 1000.0

        # ---- 1) scena
        self.f_scena.use()
        self.f_scena.clear(0, 0, 0, 1, depth=1.0)
        additivo = lambda: (ctx.enable(moderngl.BLEND), setattr(ctx, "blend_func",
                                                                (moderngl.ONE, moderngl.ONE)))
        if self.pos is not None and self.presenza > 0.001:
            n_punti, L, campo_max, pieno = self._prepara(t)
            # spessori, bagliore e sfasamento dei colori sono proporzionati al volto:
            # quando e' piccolo nell'inquadratura non devono impastarlo
            self.rapporto = float(np.clip(L / 0.45, 0.4, 1.3))
            # materializzazione: il fronte avanza lungo il campo, dal naso verso l'esterno
            p = self.presenza
            rivela = 1e9 if p >= 1.0 else (p * p * (3 - 2 * p)) * (campo_max * 1.08 + 0.05)

            ctx.enable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            # la mesh si disegna sempre (nasconde cio' che sta dietro); le linee solo se servono
            imposta(self.p_mesh, colore=colore, n_linee=float(self.n_linee), fase=self.fase,
                    spessore=self.spessore * self.ss * self.rapporto ** 0.6, energia=self.energia, tempo=tempo,
                    y_fascio=1.3 - (t * 0.45 % 1.0) * 2.6, rivela=float(rivela),
                    in_formazione=0.0 if p >= 1.0 else 1.0, voce=self.voce,
                    sfumatura=self.sfumatura,
                    tremolio=self.rabbia, mostra_linee=0.0 if self.stile == "puntini" else 1.0)
            if "impulsi" in self.p_mesh:
                self.p_mesh["impulsi"].write(self.impulsi.astype("f4").tobytes())
            self.v_mesh.render(moderngl.TRIANGLES)
            if self.corna > 0.01:
                self.v_corna.render(moderngl.TRIANGLES)

            if pieno > 0:                  # compaiono solo nel buco della bocca
                imposta(self.p_gola, colore=colore, forza=pieno, tempo=tempo)
                self.v_gola.render(moderngl.TRIANGLES, vertices=6)
                if self.denti:
                    imposta(self.p_denti, colore=colore, forza=pieno)
                    self.v_denti.render(moderngl.TRIANGLES)

            if self.ali > 0.01 and pieno > 0:   # dietro a tutto, fatte di caratteri
                additivo()
                ctx.depth_mask = False
                alt_cella = max(8.0, h * self.ss / ALI_RIGHE)
                self.t_glifi.use(0)
                self.t_sagoma.use(1)
                imposta(self.p_ali, atlante=0, sagoma=1, colore=colore, tempo=tempo,
                        forza=0.55 * pieno, cella=(alt_cella * 0.62, alt_cella),
                        n_glifi=float(len(GLIFI)), usa_sagoma=1.0 if (self.ha_sagoma and self.sfondo) else 0.0,
                        zoom=float(self.zoom), aspetto=float(self.aspetto),
                        centro=(float(self.centro[0]), float(self.centro[1])),
                        scena=(float(w * self.ss), float(h * self.ss)))
                self.v_ali.render(moderngl.TRIANGLES, vertices=12)
                ctx.depth_mask = True
                ctx.disable(moderngl.BLEND)

            additivo()
            ctx.depth_mask = False
            if self.iridi and pieno > 0:   # visibili solo attraverso l'apertura degli occhi
                imposta(self.p_iride, colore=colore, tempo=tempo)
                self.v_iridi.render(moderngl.POINTS, vertices=2)
            if self.tratti and pieno > 0 and self.stile != "puntini":
                imposta(self.p_tratti, colore=colore, forza=pieno * (0.55 + 0.5 * self.rabbia))
                self.v_tratti.render(moderngl.TRIANGLES, vertices=len(SEGMENTI) * 6)
            if self.stile != "linee":
                imposta(self.p_punti, colore=colore, rivela=float(rivela),
                        scala=L * w * self.ss / self.aspetto / 75.0 * (1.0 + 0.5 * self.energia))
                self.v_punti.render(moderngl.POINTS, vertices=n_punti)
            ctx.depth_mask = True

        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)

        # ---- 2) scia: il fotogramma precedente svanisce piano
        self.scie.reverse()
        (f_nuova, t_nuova), (_, t_vecchia) = self.scie
        f_nuova.use()
        self.t_scena.use(0)
        t_vecchia.use(1)
        imposta(self.p_scia, nuovo=0, vecchio=1, tenuta=self.scia ** (dt * 30.0))
        self.q_scia.render(moderngl.TRIANGLES)

        # ---- 3) bagliore: ogni livello sfoca il precedente a meta' risoluzione
        self._aloni, sorgente = [], t_nuova
        for (fa, ta), (fb, tb) in self.aloni:
            dim = ta.size
            fa.use()
            sorgente.use(0)
            imposta(self.p_sfoca, tex=0, passo=(1.0 / dim[0], 0.0))
            self.q_sfoca.render(moderngl.TRIANGLES)
            fb.use()
            ta.use(0)
            imposta(self.p_sfoca, tex=0, passo=(0.0, 1.0 / dim[1]))
            self.q_sfoca.render(moderngl.TRIANGLES)
            self._aloni.append(tb)
            sorgente = tb
        self._scena_finale = t_nuova
        self._uniformi_finali = dict(
            scena=0, a0=1, a1=2, a2=3, a3=4, a4=5, a5=6, webcam=7, t=t % 100.0, seme=float(rng.random()),
            glitch=((0.02 + 0.25 * self.energia + 0.2 * self.rabbia) * rng.random()
                    if self.glitch else 0.0),
            energia=self.energia, lampo=self.lampo, colpo=self.colpo,
            bagliore=self.bagliore * self.rapporto ** 0.5 * (1.0 + 0.3 * self.energia + 0.2 * self.voce),
            rapporto=self.rapporto, luce_sfondo=self.luce_sfondo,
            sfondo=1.0 if (self.sfondo and self.t_webcam is not None) else 0.0,
            dimensione=(float(w), float(h)))

    def componi(self, bersaglio, viewport):
        """Ultima passata (glitch, colori, righe) sul framebuffer indicato."""
        bersaglio.use()
        self.ctx.viewport = viewport
        self._scena_finale.use(0)
        for i in range(6):                 # lo shader ne legge sei: se sono meno, ripeto l'ultimo
            self._aloni[min(i, len(self._aloni) - 1)].use(1 + i)
        if self.t_webcam is not None:
            self.t_webcam.use(7)
        imposta(self.p_finale, **self._uniformi_finali)
        self.q_finale.render(moderngl.TRIANGLES)


class Pannello:
    """Pannello a schermo con i cursori. Si disegna con OpenCV e si mostra come texture."""

    LARG, RIGA, MARG, TESTA = 330, 42, 16, 46

    def __init__(self, ctx, olo):
        self.ctx, self.olo = ctx, olo
        self.visibile = False
        self.alt = self.TESTA + len(CURSORI) * self.RIGA + 34
        self.tex = ctx.texture((self.LARG, self.alt), 4, dtype="f1")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.prog = ctx.program(vertex_shader=VS_PANNELLO, fragment_shader=FS_PANNELLO)
        self.buf = ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.buf, "2f 2f", "in_pos", "in_uv")])
        self._disegnati = None

    def immagine(self):
        """Il pannello come immagine RGBA."""
        img = np.zeros((self.alt, self.LARG, 4), np.uint8)
        img[:] = (12, 16, 20, 215)
        chiaro, tenue = (235, 240, 245, 255), (130, 140, 150, 255)
        accento = tuple(int(v * 255) for v in
                        colorsys.hsv_to_rgb(self.olo.tinta, self.olo.saturazione, 1.0)) + (255,)
        scrivi = lambda testo, x, y, colore, scala=0.5: cv2.putText(
            img, testo, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scala, colore, 1, cv2.LINE_AA)
        scrivi("IMPOSTAZIONI", self.MARG, 28, chiaro, 0.55)
        x0, x1 = self.MARG, self.LARG - self.MARG
        for i, (etichetta, attributo, minimo, massimo) in enumerate(CURSORI):
            y = self.TESTA + i * self.RIGA
            valore = float(getattr(self.olo, attributo))
            scrivi(etichetta, x0, y + 14, chiaro)
            scrivi(f"{valore:.2f}", x1 - 44, y + 14, tenue)
            xv = int(x0 + (x1 - x0) * np.clip((valore - minimo) / (massimo - minimo), 0, 1))
            cv2.line(img, (x0, y + 27), (x1, y + 27), (60, 68, 76, 255), 3, cv2.LINE_AA)
            cv2.line(img, (x0, y + 27), (xv, y + 27), accento, 3, cv2.LINE_AA)
            cv2.circle(img, (xv, y + 27), 7, chiaro, -1, cv2.LINE_AA)
        scrivi("TAB chiude - 1-4 preset - P colore", x0, self.alt - 14, tenue, 0.42)
        return img

    def mouse(self, x, y, premuto, scala):
        """x, y in pixel della finestra (origine in alto a sinistra). Vero se il clic e' suo."""
        x, y = x / scala - 12, y / scala - 12                     # coordinate dentro il pannello
        if not (self.visibile and premuto and 0 <= x < self.LARG and 0 <= y < self.alt):
            return False
        i = int((y - self.TESTA) // self.RIGA)
        if 0 <= i < len(CURSORI):
            _, attributo, minimo, massimo = CURSORI[i]
            frazione = np.clip((x - self.MARG) / (self.LARG - 2 * self.MARG), 0.0, 1.0)
            setattr(self.olo, attributo, float(minimo + frazione * (massimo - minimo)))
        return True

    def disegna(self, fb_w, fb_h, scala):
        if not self.visibile:
            return
        valori = tuple(round(float(getattr(self.olo, c[1])), 3) for c in CURSORI)
        if valori != self._disegnati:                             # ridisegno solo se cambia qualcosa
            self._disegnati = valori
            self.tex.write(self.immagine().tobytes())
        # rettangolo in alto a sinistra, 12 pixel dal bordo
        x0, y0 = 12 * scala, 12 * scala
        x1, y1 = x0 + self.LARG * scala, y0 + self.alt * scala
        n = lambda x, y: (x / fb_w * 2 - 1, 1 - y / fb_h * 2)
        quad = [*n(x0, y0), 0, 0, *n(x1, y0), 1, 0, *n(x0, y1), 0, 1,
                *n(x0, y1), 0, 1, *n(x1, y0), 1, 0, *n(x1, y1), 1, 1]
        self.buf.write(np.array(quad, "f4").tobytes())
        self.ctx.viewport = (0, 0, fb_w, fb_h)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.tex.use(0)
        imposta(self.prog, tex=0)
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)


def carica_impostazioni(olo, percorso=FILE_IMPOSTAZIONI):
    """Rilegge le impostazioni salvate, ignorando valori mancanti o non validi."""
    try:
        with open(percorso, encoding="utf-8") as f:
            dati = json.load(f)
    except (OSError, ValueError):
        return
    limiti = {c[1]: c[2:] for c in CURSORI}
    for nome in DA_SALVARE:
        if nome not in dati:
            continue
        valore = dati[nome]
        if nome in limiti and isinstance(valore, (int, float)):
            setattr(olo, nome, float(np.clip(valore, *limiti[nome])))
        elif nome == "stile" and valore in STILI or nome == "motivo" and valore in MOTIVI:
            setattr(olo, nome, valore)
        elif isinstance(valore, bool) and isinstance(getattr(olo, nome), bool):
            setattr(olo, nome, valore)


def salva_impostazioni(olo, percorso=FILE_IMPOSTAZIONI):
    try:
        with open(percorso, "w", encoding="utf-8") as f:
            json.dump({nome: getattr(olo, nome) for nome in DA_SALVARE}, f, indent=2)
    except OSError as errore:
        print("Impostazioni non salvate:", errore)


# ================================================================ ingressi
def scarica(nome, url):
    if not os.path.exists(nome):
        print(f"Scarico {nome} (solo la prima volta)...")
        urllib.request.urlretrieve(url, nome)


class FiltroEuro:
    """Filtro "One Euro": toglie il tremolio da fermi senza aggiungere ritardo nei movimenti."""

    def __init__(self, taglio_min=1.5, beta=25.0, taglio_derivata=1.0):
        self.taglio_min, self.beta, self.taglio_d = taglio_min, beta, taglio_derivata
        self.x = self.dx = self.t = None

    @staticmethod
    def _alfa(taglio, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * taglio * dt))

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += self._alfa(self.taglio_d, dt) * ((x - self.x) / dt - self.dx)
        taglio = self.taglio_min + self.beta * np.abs(self.dx)     # piu' veloce = meno filtro
        self.x = self.x + self._alfa(taglio, dt) * (x - self.x)
        return self.x

    def azzera(self):
        self.x = None


class Tracciatore(threading.Thread):
    """Legge la webcam e traccia il volto in un thread a parte."""

    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.blocco = threading.Lock()
        self.attivo = True
        self.numero = 0                    # contatore dei fotogrammi elaborati
        self.punti = self.espressioni = self.fotogramma = None
        self.spalle = self.sagoma = None
        self.usa_corpo = True              # il thread principale lo spegne quando le ali sono spente
        self.filtro, self.filtro_spalle = FiltroEuro(), FiltroEuro(taglio_min=1.0, beta=8.0)

        scarica(MODELLO, URL_MODELLO)
        self.volto = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELLO),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True))
        try:        # corpo: spalle (per attaccare le ali) e sagoma della persona (per nasconderle dietro)
            scarica(MODELLO_CORPO, URL_CORPO)
            self.corpo = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODELLO_CORPO),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                output_segmentation_masks=True))
        except Exception as errore:
            print("Tracciamento del corpo non disponibile, stimo le spalle dal volto:", errore)
            self.corpo = None

    def _traccia_corpo(self, immagine, ms, adesso):
        """Aggiorna spalle e sagoma. Qualsiasi errore spegne il tracciamento del corpo."""
        try:
            risultato = self.corpo.detect_for_video(immagine, ms)
            spalle = None
            if risultato.pose_landmarks:
                a, b = risultato.pose_landmarks[0][11], risultato.pose_landmarks[0][12]
                if min(a.visibility, b.visibility) > 0.5:
                    spalle = np.float32(sorted([[a.x, a.y], [b.x, b.y]]))   # sinistra, destra
                    spalle = self.filtro_spalle(spalle, adesso).astype(np.float32)
            if spalle is None:
                self.filtro_spalle.azzera()
            sagoma = None
            if risultato.segmentation_masks:
                m = np.asarray(risultato.segmentation_masks[0].numpy_view(), np.float32)
                m = cv2.resize(m.reshape(m.shape[0], m.shape[1]), (256, 144),
                               interpolation=cv2.INTER_AREA)
                m = np.clip(m * 255.0, 0, 255)
                vecchia = self._sagoma_f if getattr(self, "_sagoma_f", None) is not None else m
                self._sagoma_f = vecchia * 0.4 + m * 0.6            # meno sfarfallio sui bordi
                sagoma = self._sagoma_f.astype(np.uint8)
            return spalle, sagoma
        except Exception as errore:
            print("Tracciamento del corpo disattivato:", errore)
            self.corpo = None
            return None, None

    def run(self):
        ultimo_ms, n = 0, 0
        spalle = sagoma = None
        while self.attivo:
            ok, fotogramma = self.cam.read()
            if not ok:
                time.sleep(0.01)
                continue
            fotogramma = cv2.flip(fotogramma, 1)                  # effetto specchio
            immagine = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(fotogramma, cv2.COLOR_BGR2RGB))
            adesso = time.monotonic()
            ms = max(ultimo_ms + 1, int(adesso * 1000))
            ultimo_ms = ms
            risultato = self.volto.detect_for_video(immagine, ms)

            punti = espressioni = None
            if risultato.face_landmarks:
                punti = np.array([[p.x, p.y, p.z] for p in risultato.face_landmarks[0]], np.float32)
                punti = self.filtro(punti, adesso).astype(np.float32)
                if risultato.face_blendshapes:
                    espressioni = {b.category_name: b.score
                                   for b in risultato.face_blendshapes[0]}
            else:
                self.filtro.azzera()

            # corpo: un fotogramma si' e uno no, e solo se le ali sono accese
            if self.corpo is None or not self.usa_corpo:
                spalle = sagoma = None
            elif n % 2 == 0:
                spalle, sagoma = self._traccia_corpo(immagine, ms, adesso)
            n += 1

            with self.blocco:
                self.punti, self.espressioni, self.fotogramma = punti, espressioni, fotogramma
                self.spalle, self.sagoma = spalle, sagoma
                self.numero += 1

    def leggi(self):
        with self.blocco:
            return (self.numero, self.punti, self.espressioni, self.fotogramma,
                    self.spalle, self.sagoma)


class Microfono:
    """Volume del microfono in tempo reale. Se manca sounddevice, resta a zero."""

    def __init__(self):
        self.livello = 0.0
        self.flusso = None
        try:
            import sounddevice
            self.flusso = sounddevice.InputStream(channels=1, samplerate=16000, blocksize=256,
                                                  callback=self._blocco)
            self.flusso.start()
        except Exception as errore:
            print("Reazione alla voce disattivata (pip install sounddevice):", errore)

    def _blocco(self, dati, _n, _tempo, _stato):
        self.livello = float(np.sqrt(np.mean(dati * dati)))

    def chiudi(self):
        if self.flusso:
            self.flusso.stop()
            self.flusso.close()


# ================================================================ programma
def main():
    import glfw

    cam = cv2.VideoCapture(CAMERA)
    if not cam.isOpened():
        raise SystemExit("Non riesco ad aprire la webcam: prova a cambiare CAMERA.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, primo = cam.read()
    if not ok:
        raise SystemExit("La webcam non restituisce immagini.")

    if not glfw.init():
        raise SystemExit("Impossibile inizializzare la finestra (GLFW).")
    monitor = glfw.get_primary_monitor()
    modo = glfw.get_video_mode(monitor)
    # risoluzione interna adattata al monitor; la finestra parte a due terzi dello schermo
    altezza = ALTEZZA or min(modo.size.height, 1080)
    larghezza = int(round(altezza * primo.shape[1] / primo.shape[0] / 8)) * 8
    fin_h = min(altezza, modo.size.height * 2 // 3)
    fin_w = fin_h * larghezza // altezza

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    finestra = glfw.create_window(fin_w, fin_h, "Maschera", None, None)
    if not finestra:
        raise SystemExit("La scheda video non supporta OpenGL 3.3.")
    glfw.make_context_current(finestra)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    print(f"GPU: {ctx.info['GL_RENDERER']} - rendering a {larghezza}x{altezza}")
    olo = Ologramma(ctx, larghezza, altezza)
    carica_impostazioni(olo)
    pannello = Pannello(ctx, olo)
    microfono = Microfono()

    virtuale = None
    if "--virtuale" in sys.argv:
        import pyvirtualcam
        virtuale = pyvirtualcam.Camera(width=larghezza, height=altezza, fps=30)
        print("Webcam virtuale attiva:", virtuale.device)

    stato = dict(selettore=False, adatta=True, screenshot=False, video=None, intero=None)
    alterna = {glfw.KEY_O: "iridi", glfw.KEY_D: "denti", glfw.KEY_T: "tratti",
               glfw.KEY_G: "glitch", glfw.KEY_B: "sfondo"}
    prossimo = lambda elenco, attuale: elenco[(elenco.index(attuale) + 1) % len(elenco)]

    def tasto(_finestra, codice, _scan, azione, _mod):
        if azione == glfw.RELEASE:
            return
        # frecce: funzionano anche tenendole premute
        if codice in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            olo.tinta = (olo.tinta + (0.01 if codice == glfw.KEY_RIGHT else -0.01)) % 1.0
        elif codice in (glfw.KEY_UP, glfw.KEY_DOWN):
            passo = 0.05 if codice == glfw.KEY_UP else -0.05
            olo.saturazione = float(np.clip(olo.saturazione + passo, 0.0, 1.0))
        if azione != glfw.PRESS:
            return
        if codice in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(finestra, True)
        elif codice == glfw.KEY_TAB:
            pannello.visibile = not pannello.visibile
        elif glfw.KEY_1 <= codice <= glfw.KEY_4:
            print("Preset:", olo.preset(codice - glfw.KEY_0))
        elif codice in alterna:
            setattr(olo, alterna[codice], not getattr(olo, alterna[codice]))
        elif codice == glfw.KEY_K:                                # corna: spegne e riaccende
            if olo.corna > 0.01:
                olo._corna_prima, olo.corna = olo.corna, 0.0
            else:
                olo.corna = olo._corna_prima or 1.0
        elif codice == glfw.KEY_A:                                # ali: spegne e riaccende
            if olo.ali > 0.01:
                olo._ali_prima, olo.ali = olo.ali, 0.0
            else:
                olo.ali = olo._ali_prima or 1.0
        elif codice == glfw.KEY_M:
            olo.stile = prossimo(STILI, olo.stile)
        elif codice == glfw.KEY_L:
            olo.motivo = prossimo(MOTIVI, olo.motivo)
        elif codice == glfw.KEY_Z:
            stato["adatta"] = not stato["adatta"]
        elif codice == glfw.KEY_P:
            stato["selettore"] = True
        elif codice == glfw.KEY_S:
            stato["screenshot"] = True
        elif codice == glfw.KEY_R:
            if stato["video"] is None:
                nome = time.strftime("maschera_%Y%m%d_%H%M%S.mp4")
                stato["video"] = cv2.VideoWriter(nome, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                 (larghezza, altezza))
                print("Registrazione avviata:", nome)
            else:
                stato["video"].release()
                stato["video"] = None
                print("Registrazione salvata.")
        elif codice == glfw.KEY_F:
            if stato["intero"] is None:                           # ricordo posizione e misura
                stato["intero"] = (*glfw.get_window_pos(finestra), *glfw.get_window_size(finestra))
                glfw.set_window_monitor(finestra, monitor, 0, 0, modo.size.width,
                                        modo.size.height, modo.refresh_rate)
            else:
                glfw.set_window_monitor(finestra, None, *stato["intero"], 0)
                stato["intero"] = None
            glfw.swap_interval(1)

    glfw.set_key_callback(finestra, tasto)

    def scegli_colore():
        """Apre il selettore di colore di sistema e applica la scelta."""
        try:
            import tkinter
            from tkinter import colorchooser
            radice = tkinter.Tk()
            radice.withdraw()
            radice.attributes("-topmost", True)
            attuale = colorsys.hsv_to_rgb(olo.tinta, olo.saturazione, 1.0)
            rgb, _ = colorchooser.askcolor(
                color="#%02x%02x%02x" % tuple(int(v * 255) for v in attuale),
                title="Colore della maschera", parent=radice)
            radice.destroy()
            if rgb:
                olo.tinta, olo.saturazione, _ = colorsys.rgb_to_hsv(*(v / 255 for v in rgb))
        except Exception as errore:
            print("Selettore non disponibile, usa le frecce:", errore)

    tracciatore = Tracciatore(cam)
    tracciatore.start()

    inizio = prima = time.monotonic()
    visto, ultimo_video = -1, 0.0
    while not glfw.window_should_close(finestra):
        glfw.poll_events()
        adesso = time.monotonic()
        t, dt, prima = adesso - inizio, min(adesso - prima, 1 / 20), adesso

        if stato["selettore"]:
            stato["selettore"] = False
            scegli_colore()
            prima = time.monotonic()                              # niente salto dopo la pausa

        tracciatore.usa_corpo = olo.ali > 0.01
        numero, punti, espressioni, fotogramma, spalle, sagoma = tracciatore.leggi()
        if numero != visto:                                       # c'e' un fotogramma nuovo
            visto = numero
            olo.carica_sagoma(sagoma)
            if olo.sfondo and fotogramma is not None:
                olo.carica_webcam(fotogramma)

        # con la webcam sullo sfondo la maschera deve restare sovrapposta al volto vero
        olo.adatta = stato["adatta"] and not olo.sfondo
        olo.aggiorna(punti, espressioni, dt, microfono.livello, t, spalle)
        olo.disegna(t, dt)

        # sulla finestra, mantenendo le proporzioni (bande nere se servono)
        fb_w, fb_h = glfw.get_framebuffer_size(finestra)
        scala = min(fb_w / larghezza, fb_h / altezza)
        vw, vh = int(larghezza * scala), int(altezza * scala)
        ctx.screen.use()
        ctx.viewport = (0, 0, fb_w, fb_h)
        ctx.screen.clear(0, 0, 0, 1)
        olo.componi(ctx.screen, ((fb_w - vw) // 2, (fb_h - vh) // 2, vw, vh))

        # pannello: sopra a tutto, solo sulla finestra (non finisce in video e screenshot)
        dpi = fb_w / max(1, glfw.get_window_size(finestra)[0])    # schermi ad alta densita'
        scala_ui = dpi * max(1.0, fb_h / dpi / 1080)              # piu' grande sui monitor grandi
        mx, my = glfw.get_cursor_pos(finestra)
        premuto = glfw.get_mouse_button(finestra, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        pannello.mouse(mx * dpi, my * dpi, premuto, scala_ui)
        pannello.disegna(fb_w, fb_h, scala_ui)
        glfw.swap_buffers(finestra)

        # uscite facoltative: screenshot, video (30 fotogrammi al secondo), webcam virtuale
        registra = stato["video"] is not None and adesso - ultimo_video >= 1 / 30
        if stato["screenshot"] or registra or virtuale:
            immagine = np.ascontiguousarray(olo.leggi_immagine())
            if stato["screenshot"]:
                stato["screenshot"] = False
                nome = time.strftime("maschera_%Y%m%d_%H%M%S.png")
                cv2.imwrite(nome, immagine)
                print("Screenshot salvato:", nome)
            if registra:
                ultimo_video = adesso
                stato["video"].write(immagine)
            if virtuale:
                virtuale.send(immagine[:, :, ::-1])

    salva_impostazioni(olo)
    tracciatore.attivo = False
    tracciatore.join(timeout=1.0)
    if stato["video"] is not None:
        stato["video"].release()
    microfono.chiudi()
    cam.release()
    if virtuale:
        virtuale.close()
    glfw.terminate()


if __name__ == "__main__":
    main()