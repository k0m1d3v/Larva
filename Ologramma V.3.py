"""
Maschera olografica in tempo reale dalla webcam (v5, rendering su GPU).

Installazione:
    pip install mediapipe opencv-python numpy moderngl glfw
    pip install pyvirtualcam        (solo per la webcam virtuale)

Avvio:
    python maschera.py              finestra normale
    python maschera.py --virtuale   in piu' manda l'immagine a una webcam virtuale
                                    (Windows/macOS: serve OBS installato;
                                     Linux: serve il modulo v4l2loopback)

Tasti:
    m  cambia stile (ologramma / solo puntini / solo linee)
    z  volto centrato e ingrandito <-> posizione reale
    o  iridi accese/spente
    d  denti accesi/spenti
    g  glitch acceso/spento
    b  sfondo nero <-> webcam scurita
    p  apre il selettore di colore di sistema
    freccia sinistra / destra   cambia la tinta
    freccia su / giu'           colore piu' / meno saturo
    q  esci

Come funziona:
    - un thread legge la webcam e traccia il volto (MediaPipe, su CPU);
    - il thread principale disegna con OpenGL alla frequenza dello schermo:
      mesh 3D del volto con linee di livello calcolate pixel per pixel,
      particelle, scie, bloom e glitch sono tutti shader.
"""

import colorsys
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
ALTEZZA = 720         # risoluzione verticale del rendering
N_LINEE = 30          # quante linee di livello sul volto
SPESSORE = 1.1        # spessore delle linee, in pixel
BAGLIORE = 1.0        # intensita' dell'alone luminoso
TINTA = 0.50          # colore iniziale: 0 rosso, 0.33 verde, 0.5 ciano, 0.66 blu, 0.83 magenta
SATURAZIONE = 0.85    # 0 = bianco, 1 = colore pieno
GRANDEZZA = 0.64      # con 'z' attivo: altezza del volto rispetto alla finestra
SCIA = 0.80           # persistenza della scia (0 = nessuna, 0.95 = lunghissima)
MODELLO = "face_landmarker.task"
URL_MODELLO = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
STILI = ["ologramma", "puntini", "linee"]

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


TRIANGOLI = _triangoli()


def _suddividi(tri):
    """Divide ogni triangolo in quattro: restituisce gli spigoli e i nuovi triangoli."""
    spigoli = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1), axis=0)
    indice = {tuple(s): 478 + i for i, s in enumerate(spigoli)}
    meta = lambda a, b: indice[(min(a, b), max(a, b))]
    nuovi = []
    for a, b, c in tri:
        ab, bc, ca = meta(a, b), meta(b, c), meta(c, a)
        nuovi += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    return spigoli.astype(np.int32), np.array(nuovi, np.int32)


SPIGOLI, TRI_FINI = _suddividi(TRIANGOLI)
COLONNE_DENTI = 33    # suddivisioni di ogni arcata di denti
N_VERTICI = 478 + len(SPIGOLI)
rng = np.random.default_rng()


# ================================================================ shader
VS_MESH = """
#version 330
in vec3 in_pos;
in float in_campo;
in vec3 in_n;
out float v_campo;
out vec3 v_n;
out vec2 v_ndc;
void main() {
    gl_Position = vec4(in_pos, 1.0);
    v_campo = in_campo;
    v_n = in_n;
    v_ndc = in_pos.xy;
}
"""

FS_MESH = """
#version 330
uniform vec3 colore;
uniform float n_linee, fase, spessore, presenza, energia, y_fascio, mostra_linee, tempo;
in float v_campo;
in vec3 v_n;
in vec2 v_ndc;
out vec4 f;

float caso(float x) { return fract(sin(x * 12.9898) * 43758.5453); }

void main() {
    float fronte = clamp(-normalize(v_n).z, 0.0, 1.0);        // 1 = rivolto alla camera

    // linee di livello della profondita' del volto, spessore costante in pixel
    float l = n_linee * pow(max(v_campo, 1e-4), 0.75) - fase;
    float pendenza = max(fwidth(l), 1e-5);
    float numero = floor(l + 0.5);                             // quale linea e' questa
    float sp = spessore * (0.65 + 0.7 * fronte);               // piu' spesse verso la camera
    float d = abs(fract(l - 0.5) - 0.5) / pendenza;
    float linea = 1.0 - smoothstep(sp - 0.6, sp + 0.6, d);

    // linee secondarie sottili tra una principale e l'altra, solo dove c'e' spazio
    float d3 = abs(fract(l * 3.0 - 0.5) - 0.5) / (pendenza * 3.0);
    float sottili = (1.0 - smoothstep(0.2, 0.9, d3)) * 0.20 * (1.0 - smoothstep(0.08, 0.20, pendenza));

    // ogni linea respira per conto suo; un'onda di luce corre dal naso verso l'esterno
    float respiro = 0.75 + 0.25 * sin(tempo * (1.5 + 3.0 * caso(numero)) + 6.283 * caso(numero + 7.0));
    float onda = exp(-pow((fract(l / 9.0 - tempo * 0.22) - 0.5) * 5.0, 2.0));

    // dove le linee sarebbero piu' fitte dei pixel le sostituisco con una luce uniforme
    float fitte = smoothstep(0.35, 0.60, pendenza);
    float luce = mix(linea * respiro * (1.0 + (0.8 + 0.6 * energia) * onda), 0.35, fitte) + sottili;
    luce *= mostra_linee;

    float bordo = pow(1.0 - fronte, 3.0) * mostra_linee;      // alone sul profilo
    float fascio = exp(-pow((v_ndc.y - y_fascio) / 0.035, 2.0));

    vec3 c = colore * (luce * (0.30 + 0.95 * fronte) + bordo * 0.30 + 0.012 + 0.03 * fronte * fronte);
    c += vec3(1.0) * linea * onda * fronte * 0.12 * mostra_linee;   // le creste dell'onda sbiancano
    c += colore * fascio * (0.04 + 0.9 * linea * mostra_linee);
    f = vec4(c * presenza * (1.0 + 0.25 * energia), 1.0);
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
uniform float presenza;
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
    f = vec4((colore * 0.6 + 0.4) * luce * presenza, 1.0);
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

VS_PUNTI = """
#version 330
uniform float scala;
in vec3 in_pos;
in float in_lum;
out float v_lum;
void main() {
    gl_Position = vec4(in_pos.xy, in_pos.z - 0.004, 1.0);
    gl_PointSize = scala * (0.6 + in_lum);
    v_lum = in_lum;
}
"""

FS_PUNTI = """
#version 330
uniform vec3 colore;
uniform float forza;
in float v_lum;
out vec4 f;
void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.0, d);
    f = vec4((colore * 0.7 + 0.3) * a * a * v_lum * forza, 1.0);
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

FS_FINALE = """
#version 330
uniform sampler2D scena, alone1, alone2, webcam;
uniform float t, seme, glitch, energia, sfondo, bagliore;
uniform vec2 dimensione;
in vec2 uv;
out vec4 f;

float caso(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec3 luce(vec2 p) {
    return texture(scena, p).rgb
         + bagliore * (0.55 * texture(alone1, p).rgb + 0.95 * texture(alone2, p).rgb);
}

void main() {
    vec2 p = uv;
    // glitch: alcune bande orizzontali scivolano di lato
    float banda = floor(p.y * 28.0 + seme * 7.0);
    if (caso(vec2(banda, seme)) < glitch)
        p.x += (caso(vec2(seme, banda)) - 0.5) * 0.14;

    // aberrazione cromatica, piu' forte verso i bordi e quando apri la bocca
    float ab = (0.0012 + 0.006 * energia) * (0.5 + length(uv - 0.5) * 2.0);
    vec3 c = vec3(luce(p + vec2(ab, 0.0)).r, luce(p).g, luce(p - vec2(ab, 0.0)).b);

    c = 1.0 - exp(-c * 1.5);                                   // compressione dei toni
    c *= 0.86 + 0.14 * sin(uv.y * dimensione.y * 3.14159);     // righe di scansione
    c *= 1.0 - 0.45 * pow(length(uv - 0.5) * 1.3, 2.5);        // vignettatura
    c += (caso(uv * dimensione + t) - 0.5) * 0.035;            // grana
    c *= 0.94 + 0.06 * caso(vec2(t, 3.0));                     // sfarfallio

    vec3 fondo = texture(webcam, vec2(uv.x, 1.0 - uv.y)).bgr * 0.22 * sfondo;
    f = vec4(fondo + c, 1.0);
}
"""


# ================================================================ rendering
def imposta(programma, **valori):
    """Assegna gli uniform ignorando quelli che lo shader non usa."""
    for nome, valore in valori.items():
        if nome in programma:
            programma[nome].value = valore


class Ologramma:
    """Stato dell'effetto (particelle, espressioni) e tutte le passate di rendering."""

    def __init__(self, ctx, larghezza, altezza):
        self.ctx = ctx
        self.dim = (larghezza, altezza)
        self.aspetto = larghezza / altezza

        # ---- stato
        self.pos = None                   # (478, 3) coordinate normalizzate 0..1 (z come x)
        self.vel = None
        self.presenza = 0.0               # 0 = volto assente, 1 = agganciato
        self.energia = self.sorriso = self.sopracciglia = 0.0
        self.fase = 0.0
        self.tinta, self.saturazione = TINTA, SATURAZIONE
        self.occhi = np.ones(2, np.float32)          # apertura di ciascun occhio (0 = chiuso)
        # denti: non sono tracciati, li ricostruisco. Le due misure (in larghezze di volto)
        # si tarano da sole ogni volta che la bocca e' chiusa.
        self.dente_su = 0.125             # dalla base del naso alla linea delle labbra
        self.dente_giu = 0.320            # dalla linea delle labbra al mento
        self.adatta = True                           # volto centrato e ingrandito
        self.centro = np.array([self.aspetto / 2, 0.5, 0.0], np.float32)
        self.zoom = 1.0

        # ---- programmi
        P = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_mesh = P(VS_MESH, FS_MESH)
        self.p_punti = P(VS_PUNTI, FS_PUNTI)
        self.p_iride = P(VS_IRIDE, FS_IRIDE)
        self.p_denti = P(VS_DENTI, FS_DENTI)
        self.p_scia = P(VS_SCHERMO, FS_SCIA)
        self.p_sfoca = P(VS_SCHERMO, FS_SFOCA)
        self.p_finale = P(VS_SCHERMO, FS_FINALE)

        # ---- geometria
        self.b_mesh = ctx.buffer(reserve=N_VERTICI * 7 * 4, dynamic=True)
        self.b_punti = ctx.buffer(reserve=N_VERTICI * 4 * 4, dynamic=True)
        indici = ctx.buffer(TRI_FINI.astype("i4").tobytes())
        quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self.v_mesh = ctx.vertex_array(
            self.p_mesh, [(self.b_mesh, "3f 1f 3f", "in_pos", "in_campo", "in_n")], indici)
        self.v_punti = ctx.vertex_array(
            self.p_punti, [(self.b_punti, "3f 1f", "in_pos", "in_lum")])
        k = np.arange(COLONNE_DENTI - 1)
        quadri = np.stack([2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2], 1)
        quadri = np.vstack([quadri, quadri + 2 * COLONNE_DENTI])        # due arcate
        self.b_denti = ctx.buffer(reserve=4 * COLONNE_DENTI * 7 * 4, dynamic=True)
        self.v_denti = ctx.vertex_array(
            self.p_denti, [(self.b_denti, "3f 2f 2f", "in_pos", "in_uv", "in_par")],
            ctx.buffer(quadri.astype("i4").tobytes()))
        self.b_iridi = ctx.buffer(reserve=2 * 5 * 4, dynamic=True)
        self.v_iridi = ctx.vertex_array(
            self.p_iride, [(self.b_iridi, "3f 1f 1f", "in_pos", "in_dim", "in_aperto")])
        V = lambda prog: ctx.vertex_array(prog, [(quad, "2f", "in_pos")])
        self.q_scia = V(self.p_scia)
        self.q_sfoca, self.q_finale = V(self.p_sfoca), V(self.p_finale)

        # ---- framebuffer (HDR a 16 bit)
        def bersaglio(dim, profondita=False):
            tex = ctx.texture(dim, 4, dtype="f2")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            dep = ctx.depth_renderbuffer(dim) if profondita else None
            return ctx.framebuffer([tex], dep), tex

        self.f_scena, self.t_scena = bersaglio(self.dim, True)
        self.scie = [bersaglio(self.dim), bersaglio(self.dim)]
        meta = (larghezza // 2, altezza // 2)
        ottavo = (larghezza // 8, altezza // 8)
        self.alone1 = [bersaglio(meta), bersaglio(meta)]
        self.alone2 = [bersaglio(ottavo), bersaglio(ottavo)]

        self.t_webcam = None

    # ------------------------------------------------------------ dati in ingresso
    def carica_webcam(self, fotogramma):
        h, w = fotogramma.shape[:2]
        if self.t_webcam is None or self.t_webcam.size != (w, h):
            self.t_webcam = self.ctx.texture((w, h), 3, dtype="f1")
        self.t_webcam.write(np.ascontiguousarray(fotogramma).tobytes())

    # ------------------------------------------------------------ fisica
    def aggiorna(self, punti, espressioni, dt):
        e = espressioni or {}
        k = min(1.0, dt * 9.0)
        self.energia += k * (e.get("jawOpen", 0.0) - self.energia)
        sorriso = (e.get("mouthSmileLeft", 0.0) + e.get("mouthSmileRight", 0.0)) / 2
        self.sorriso += k * (sorriso - self.sorriso)
        self.sopracciglia += k * (e.get("browInnerUp", 0.0) - self.sopracciglia)
        for i, nome in enumerate(("eyeBlinkRight", "eyeBlinkLeft")):
            aperto = 1.0 - float(np.clip(e.get(nome, 0.0) * 1.6, 0, 1))
            self.occhi[i] += min(1.0, dt * 25.0) * (aperto - self.occhi[i])
        self.fase += dt * (0.35 + 1.5 * self.sopracciglia + 1.0 * self.energia)

        if punti is not None:
            if self.pos is None:
                self.pos, self.vel = punti.copy(), np.zeros_like(punti)
            # molla smorzata: la maschera insegue il volto con un filo di inerzia
            self.vel += (260.0 * (punti - self.pos) - 24.0 * self.vel) * dt
            self.pos += self.vel * dt
            self.presenza = min(1.0, self.presenza + dt * 3.0)

            # taratura dei denti: a bocca chiusa misuro dove si incontrano le labbra
            U = self._unita(self.pos)
            _, giu, _, L = self._terna(U)
            if (U[14] - U[13]) @ giu / L < 0.02 and self.energia < 0.08:
                k = min(1.0, dt * 2.0)
                self.dente_su += k * ((U[13] - U[2]) @ giu / L - self.dente_su)
                self.dente_giu += k * ((U[152] - U[14]) @ giu / L - self.dente_giu)

            # inquadratura: centro e zoom inseguono il volto (o tornano alla posizione reale)
            if self.adatta:
                alto, basso = self._unita(self.pos[10]), self._unita(self.pos[152])
                centro = (alto + basso) / 2
                zoom = GRANDEZZA / (np.linalg.norm(alto - basso) + 1e-6)
            else:
                centro, zoom = np.array([self.aspetto / 2, 0.5, 0.0], np.float32), 1.0
            k = min(1.0, dt * 6.0)
            self.centro += k * (centro - self.centro)
            self.zoom += k * (zoom - self.zoom)
        elif self.pos is not None:
            # volto perso: i punti si sciolgono verso l'alto
            self.vel = self.vel * (1.0 - 0.8 * dt) + rng.normal(0, 0.9 * dt, self.vel.shape)
            self.vel[:, 1] -= 0.25 * dt
            self.vel[:, 2] = 0.0
            self.pos += self.vel.astype(np.float32) * dt
            self.presenza = max(0.0, self.presenza - dt * 0.8)

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

    def _prepara_mesh(self):
        """Calcola per ogni punto: posizione su schermo, profondita' locale e normale."""
        p = self.pos
        P = (self._unita(p) - self.centro) * self.zoom + (self.aspetto / 2, 0.5, 0.0)

        # terna solidale con la testa: le linee restano incollate al volto quando ruoti
        destra, giu, avanti, larghezza = self._terna(P)
        self._prepara_denti(P, destra, giu, avanti, larghezza)

        # normali per vertice, orientate verso l'esterno della testa
        a, b, c = P[TRIANGOLI[:, 0]], P[TRIANGOLI[:, 1]], P[TRIANGOLI[:, 2]]
        nf = np.cross(b - a, c - a)
        centro_testa = P.mean(axis=0) - avanti * larghezza * 0.6
        nf[np.einsum("ij,ij->i", nf, (a + b + c) / 3 - centro_testa) < 0] *= -1
        nv = np.zeros_like(P)
        for k in range(3):
            np.add.at(nv, TRIANGOLI[:, k], nf)
        nv /= np.linalg.norm(nv, axis=1, keepdims=True) + 1e-9

        # suddivisione con superficie arrotondata (Phong tessellation): ogni punto medio
        # viene spinto verso i piani tangenti dei due estremi, cosi' le linee non fanno spigoli
        ia, ib = SPIGOLI[:, 0], SPIGOLI[:, 1]
        medio = (P[ia] + P[ib]) / 2
        su_a = medio - np.einsum("ij,ij->i", medio - P[ia], nv[ia])[:, None] * nv[ia]
        su_b = medio - np.einsum("ij,ij->i", medio - P[ib], nv[ib])[:, None] * nv[ib]
        medio += 0.75 * ((su_a + su_b) / 2 - medio)
        n_medio = nv[ia] + nv[ib]
        n_medio /= np.linalg.norm(n_medio, axis=1, keepdims=True) + 1e-9
        iridi = P[[468, 473]], [np.linalg.norm(P[469:473, :2] - P[468, :2], axis=1).mean(),
                                np.linalg.norm(P[474:478, :2] - P[473, :2], axis=1).mean()]
        P, nv = np.vstack([P, medio]), np.vstack([nv, n_medio])

        campo = np.clip((P - P[1]) @ -avanti / larghezza, 0.0, None)   # 0 sulla punta del naso
        in_ndc = lambda Q: np.stack([Q[:, 0] / self.aspetto * 2 - 1, 1 - Q[:, 1] * 2,
                                     np.clip(Q[:, 2] / self.aspetto * 2, -0.98, 0.98)], axis=1)
        ndc = in_ndc(P)

        # iridi: un disco luminoso per occhio, appena dietro le palpebre
        centri, raggi = iridi
        dati = np.zeros((2, 5), np.float32)
        dati[:, :3] = in_ndc(centri) + (0.0, 0.0, 0.012)
        dati[:, 3] = np.array(raggi) * self.dim[1] * 2 * 1.5       # diametro in pixel (+margine)
        dati[:, 4] = self.occhi * self.presenza ** 3
        self.b_iridi.write(dati.tobytes())
        self.b_mesh.write(np.hstack([ndc, campo[:, None], nv]).astype("f4").tobytes())

        # puntini: uno per ogni vertice della mesh suddivisa
        z = ndc[:, 2]
        vicinanza = 1.0 - (z - z.min()) / (np.ptp(z) + 1e-6)
        lum = (0.25 + 0.75 * vicinanza) * self.presenza
        self.b_punti.write(np.hstack([ndc, lum[:, None]]).astype("f4").tobytes())
        return len(ndc), larghezza / self.aspetto

    def _prepara_denti(self, P, destra, giu, avanti, L):
        """Due arcate di denti, visibili solo attraverso l'apertura della bocca."""
        dietro = -avanti
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
            riga = lambda Q, v: np.hstack([Q, x / larg_dente, np.full_like(x, v),
                                           np.abs(x) / 0.27, par])
            return np.stack([riga(bordo, 1.0), riga(gengiva, -0.45)], 1).reshape(-1, 7)

        # superiori: fissi rispetto al cranio (ancorati alla base del naso)
        sopra = arcata(P[2], self.dente_su + 0.012, +1, 0.075, 0.054,
                       (P[13] - P[2]) @ dietro + 0.02 * L)
        # inferiori: seguono la mandibola (ancorati al mento)
        sotto = arcata(P[152], -self.dente_giu, -1, 0.060, 0.042,
                       (P[14] - P[152]) @ dietro + 0.03 * L)
        dati = np.vstack([sopra, sotto]).astype(np.float32)
        # profondita' quasi al fondo: passano il test solo dove la mesh del volto ha un buco
        dati[:, :3] = np.stack([dati[:, 0] / self.aspetto * 2 - 1, 1 - dati[:, 1] * 2,
                                np.full(len(dati), 0.999, np.float32)], 1)
        self.b_denti.write(dati.tobytes())

    # ------------------------------------------------------------ disegno
    def disegna(self, t, dt, stile="ologramma", glitch=True, sfondo=False, iridi=True,
                denti=True):
        ctx, (w, h) = self.ctx, self.dim
        # il colore scelto; sorridendo la tinta scivola di un terzo di giro
        tinta = (self.tinta + 0.33 * float(np.clip(self.sorriso, 0, 1))) % 1.0
        colore = colorsys.hsv_to_rgb(tinta, self.saturazione, 1.0)

        # ---- 1) scena: mesh con linee e puntini
        self.f_scena.use()
        self.f_scena.clear(0, 0, 0, 1, depth=1.0)
        additivo = lambda: (ctx.enable(moderngl.BLEND), setattr(ctx, "blend_func",
                                                                (moderngl.ONE, moderngl.ONE)))
        visibile = self.pos is not None and self.presenza > 0.001

        if visibile:
            n_punti, larghezza = self._prepara_mesh()
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.disable(moderngl.BLEND)
            # la mesh si disegna sempre (nasconde cio' che sta dietro); le linee solo se servono
            imposta(self.p_mesh, colore=colore, n_linee=float(N_LINEE), fase=self.fase,
                    spessore=SPESSORE, presenza=self.presenza ** 3, energia=self.energia,
                    y_fascio=1.3 - (t * 0.45 % 1.0) * 2.6,
                    mostra_linee=0.0 if stile == "puntini" else 1.0, tempo=t % 1000.0)
            self.v_mesh.render(moderngl.TRIANGLES)

            if denti:                      # compaiono solo nel buco della bocca
                imposta(self.p_denti, colore=colore, presenza=self.presenza ** 3)
                self.v_denti.render(moderngl.TRIANGLES)

            if iridi:                      # visibili solo attraverso l'apertura degli occhi
                additivo()
                ctx.depth_mask = False
                imposta(self.p_iride, colore=colore, tempo=t % 1000.0)
                self.v_iridi.render(moderngl.POINTS, vertices=2)
                ctx.depth_mask = True

            if stile != "linee":
                additivo()
                ctx.depth_mask = False
                imposta(self.p_punti, colore=colore, forza=0.9,
                        scala=larghezza * w / 75.0 * (1.0 + 0.5 * self.energia))
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
        imposta(self.p_scia, nuovo=0, vecchio=1, tenuta=SCIA ** (dt * 30.0))
        self.q_scia.render(moderngl.TRIANGLES)

        # ---- 3) alone su due scale (sfocatura separabile a bassa risoluzione)
        def sfoca(coppia, sorgente, giri):
            (fa, ta), (fb, tb) = coppia
            dim = ta.size
            for i in range(giri):
                fa.use()
                (sorgente if i == 0 else tb).use(0)
                imposta(self.p_sfoca, tex=0, passo=(1.0 / dim[0], 0.0))
                self.q_sfoca.render(moderngl.TRIANGLES)
                fb.use()
                ta.use(0)
                imposta(self.p_sfoca, tex=0, passo=(0.0, 1.0 / dim[1]))
                self.q_sfoca.render(moderngl.TRIANGLES)
            return tb

        self._alone1 = sfoca(self.alone1, t_nuova, 1)
        self._alone2 = sfoca(self.alone2, self._alone1, 2)
        self._scena_finale = t_nuova
        self._uniformi_finali = dict(
            scena=0, alone1=1, alone2=2, webcam=3, t=t % 100.0, seme=float(rng.random()),
            glitch=(0.02 + 0.25 * self.energia) * rng.random() if glitch else 0.0,
            energia=self.energia, bagliore=BAGLIORE * (1.0 + 0.3 * self.energia),
            sfondo=1.0 if (sfondo and self.t_webcam is not None) else 0.0,
            dimensione=(float(w), float(h)))

    def componi(self, bersaglio, viewport):
        """Ultima passata (glitch, colori, righe) sul framebuffer indicato."""
        bersaglio.use()
        self.ctx.viewport = viewport
        self._scena_finale.use(0)
        self._alone1.use(1)
        self._alone2.use(2)
        if self.t_webcam is not None:
            self.t_webcam.use(3)
        imposta(self.p_finale, **self._uniformi_finali)
        self.q_finale.render(moderngl.TRIANGLES)


# ================================================================ tracciamento
def scarica(nome, url):
    if not os.path.exists(nome):
        print(f"Scarico {nome} (solo la prima volta)...")
        urllib.request.urlretrieve(url, nome)


class Tracciatore(threading.Thread):
    """Legge la webcam e traccia il volto in un thread a parte."""

    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.blocco = threading.Lock()
        self.attivo = True
        self.numero = 0                    # contatore dei fotogrammi elaborati
        self.punti = self.espressioni = self.fotogramma = None

        scarica(MODELLO, URL_MODELLO)
        self.volto = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELLO),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True))

    def run(self):
        ultimo_ms = 0
        while self.attivo:
            ok, fotogramma = self.cam.read()
            if not ok:
                time.sleep(0.01)
                continue
            fotogramma = cv2.flip(fotogramma, 1)                  # effetto specchio
            immagine = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(fotogramma, cv2.COLOR_BGR2RGB))
            ms = max(ultimo_ms + 1, int(time.monotonic() * 1000))
            ultimo_ms = ms
            risultato = self.volto.detect_for_video(immagine, ms)

            punti = espressioni = None
            if risultato.face_landmarks:
                punti = np.array([[p.x, p.y, p.z] for p in risultato.face_landmarks[0]], np.float32)
                if risultato.face_blendshapes:
                    espressioni = {b.category_name: b.score
                                   for b in risultato.face_blendshapes[0]}

            with self.blocco:
                self.punti, self.espressioni, self.fotogramma = punti, espressioni, fotogramma
                self.numero += 1

    def leggi(self):
        with self.blocco:
            return self.numero, self.punti, self.espressioni, self.fotogramma


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
    altezza = ALTEZZA
    larghezza = int(round(altezza * primo.shape[1] / primo.shape[0] / 8)) * 8

    if not glfw.init():
        raise SystemExit("Impossibile inizializzare la finestra (GLFW).")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    finestra = glfw.create_window(larghezza, altezza, "Maschera", None, None)
    if not finestra:
        raise SystemExit("La scheda video non supporta OpenGL 3.3.")
    glfw.make_context_current(finestra)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    print("GPU:", ctx.info["GL_RENDERER"])
    olo = Ologramma(ctx, larghezza, altezza)

    virtuale = uscita = None
    if "--virtuale" in sys.argv:
        import pyvirtualcam
        virtuale = pyvirtualcam.Camera(width=larghezza, height=altezza, fps=30)
        uscita = ctx.framebuffer([ctx.texture((larghezza, altezza), 3)])
        print("Webcam virtuale attiva:", virtuale.device)

    opzioni = dict(stile=0, glitch=True, sfondo=False, selettore=False, adatta=True, iridi=True,
                   denti=True)

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
        if codice == glfw.KEY_P:
            opzioni["selettore"] = True
        if codice in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(finestra, True)
        elif codice == glfw.KEY_M:
            opzioni["stile"] = (opzioni["stile"] + 1) % len(STILI)
        elif codice == glfw.KEY_Z:
            opzioni["adatta"] = not opzioni["adatta"]
        elif codice == glfw.KEY_D:
            opzioni["denti"] = not opzioni["denti"]
        elif codice == glfw.KEY_O:
            opzioni["iridi"] = not opzioni["iridi"]
        elif codice == glfw.KEY_G:
            opzioni["glitch"] = not opzioni["glitch"]
        elif codice == glfw.KEY_B:
            opzioni["sfondo"] = not opzioni["sfondo"]

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
    visto = -1
    while not glfw.window_should_close(finestra):
        glfw.poll_events()
        adesso = time.monotonic()
        t, dt, prima = adesso - inizio, min(adesso - prima, 1 / 20), adesso

        if opzioni["selettore"]:
            opzioni["selettore"] = False
            scegli_colore()
            prima = time.monotonic()                              # niente salto dopo la pausa

        numero, punti, espressioni, fotogramma = tracciatore.leggi()
        if numero != visto:                                       # c'e' un fotogramma nuovo
            visto = numero
            if opzioni["sfondo"] and fotogramma is not None:
                olo.carica_webcam(fotogramma)

        # con la webcam sullo sfondo la maschera deve restare sovrapposta al volto vero
        olo.adatta = opzioni["adatta"] and not opzioni["sfondo"]
        olo.aggiorna(punti, espressioni, dt)
        olo.disegna(t, dt, STILI[opzioni["stile"]], opzioni["glitch"], opzioni["sfondo"],
                    opzioni["iridi"], opzioni["denti"])
        olo.componi(ctx.screen, (0, 0, *glfw.get_framebuffer_size(finestra)))
        glfw.swap_buffers(finestra)

        if virtuale:
            olo.componi(uscita, (0, 0, larghezza, altezza))
            dati = np.frombuffer(uscita.read(components=3), np.uint8)
            virtuale.send(dati.reshape(altezza, larghezza, 3)[::-1])

    tracciatore.attivo = False
    tracciatore.join(timeout=1.0)
    cam.release()
    if virtuale:
        virtuale.close()
    glfw.terminate()


if __name__ == "__main__":
    main()