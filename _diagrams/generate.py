"""Generate the LLM-from-scratch diagrams as theme-aware inline SVG (styled by assets/css/main.css).

Run: python3 _diagrams/generate.py   (writes _includes/diagrams/*.svg)
"""
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "_includes" / "diagrams"


class SVG:
    def __init__(self, pid, w, h, label):
        self.pid, self.w, self.h, self.label = pid, w, h, label
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def box(self, x, y, w, h, cls, lines=(), rx=6, tcls="d-t", extra=""):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"{extra}/>')
        if isinstance(lines, str):
            lines = [lines]
        n = len(lines)
        for i, ln in enumerate(lines):
            ty = y + h / 2 + (i - (n - 1) / 2) * 16 + 4.5
            self.text(x + w / 2, ty, ln, tcls)

    def text(self, x, y, s, cls="d-t", anchor="middle", raw=False):
        body = s if raw else escape(s)
        self.add(f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" class="{cls}">{body}</text>')

    def path(self, pts, arrow=True, cls="d-line"):
        d = "M" + " L".join(f"{x:g},{y:g}" for x, y in pts)
        m = f' marker-end="url(#{self.pid}-arr)"' if arrow else ""
        self.add(f'<path d="{d}" class="{cls}"{m}/>')

    def op(self, cx, cy, kind="plus", r=8):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" class="d-op"/>')
        a = r * 0.55
        if kind == "plus":
            self.add(f'<path d="M{cx-a},{cy} L{cx+a},{cy} M{cx},{cy-a} L{cx},{cy+a}" class="d-line"/>')
        else:
            b = a * 0.75
            self.add(f'<path d="M{cx-b},{cy-b} L{cx+b},{cy+b} M{cx-b},{cy+b} L{cx+b},{cy-b}" class="d-line"/>')

    def dot(self, x, y):
        self.add(f'<circle cx="{x}" cy="{y}" r="2.6" class="d-dot"/>')

    def render(self):
        head = (
            f'<svg class="d-svg" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(self.label)}">'
            f'<defs><marker id="{self.pid}-arr" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="8" '
            f'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L10,5 L0,9 z" class="d-head"/></marker></defs>'
        )
        return "\n".join([head, *self.parts, "</svg>"]) + "\n"

    def save(self, name):
        (OUT / name).write_text(self.render())


# ---------------------------------------------------------------------------
# 1. Qwen3 4B vs Llama 3.2 1B
# ---------------------------------------------------------------------------
def llm_architectures():
    s = SVG("llm", 960, 640, "Decoder-only architectures of Qwen3 4B and Llama 3.2 1B, with the SwiGLU feed-forward layer shown in detail")

    def stack(cx, title, accent, layers, qk_norm):
        s.text(cx, 34, title, f"d-title {accent}")
        s.text(cx, 70, "Output probabilities over the vocabulary", "d-s")
        s.box(cx - 90, 96, 180, 28, "b-out", "Linear output layer")
        s.path([(cx, 96), (cx, 78)])
        s.box(cx - 90, 146, 180, 26, "b-norm", "RMSNorm")
        s.path([(cx, 146), (cx, 124)])
        s.add(f'<rect x="{cx-118}" y="188" width="236" height="292" rx="10" class="d-group {accent}"/>')
        s.text(cx + 112, 206, f"× {layers} layers", "d-s", "end")
        s.path([(cx, 206), (cx, 172)])
        s.op(cx, 214)
        s.box(cx - 90, 234, 180, 30, "b-ffn", "Feed-forward")
        s.path([(cx, 234), (cx, 222)])
        s.box(cx - 90, 278, 180, 26, "b-norm", "RMSNorm")
        s.path([(cx, 278), (cx, 264)])
        s.path([(cx, 322), (cx, 304)])
        s.dot(cx, 313)
        s.path([(cx, 313), (cx + 104, 313), (cx + 104, 214), (cx + 8, 214)])
        s.op(cx, 330)
        s.box(cx - 90, 346, 180, 64, "b-attn", ["Masked grouped-query", "attention"])
        s.path([(cx, 346), (cx, 338)])
        s.box(cx - 90, 428, 180, 26, "b-norm", "RMSNorm")
        s.path([(cx, 428), (cx, 410)])
        s.box(cx - 90, 506, 180, 28, "b-embed", "Embedding layer")
        s.path([(cx, 506), (cx, 454)])
        s.dot(cx, 468)
        s.path([(cx, 468), (cx + 104, 468), (cx + 104, 330), (cx + 8, 330)])
        s.box(cx - 90, 562, 180, 28, "b-plain", "Tokenized text")
        s.path([(cx, 562), (cx, 534)])
        s.text(cx, 630, "Input text", "d-s")
        s.path([(cx, 614), (cx, 590)])
        # Positional information and (Qwen3 only) query/key normalisation feed the attention block.
        s.box(cx - 224, 352, 96, 24, "b-pos", "RoPE", tcls="d-t d-sm")
        s.path([(cx - 128, 364), (cx - 90, 364)])
        if qk_norm:
            s.box(cx - 224, 384, 96, 24, "b-norm d-new", "Q/K RMSNorm", tcls="d-t d-sm")
            s.path([(cx - 128, 396), (cx - 90, 396)])

    stack(240, "Qwen3 4B", "acc-a", 36, True)
    stack(600, "Llama 3.2 1B", "acc-b", 16, False)

    # SwiGLU detail, shared by both models.
    s.path([(690, 249), (772, 300)], arrow=False, cls="d-leader")
    s.add('<rect x="772" y="206" width="176" height="232" rx="10" class="d-inset"/>')
    s.path([(860, 250), (860, 222)])
    s.box(830, 250, 60, 24, "b-out", "Linear", tcls="d-t d-sm")
    s.op(860, 298, "times")
    s.path([(860, 290), (860, 274)])
    s.box(782, 314, 60, 24, "b-ffn", "SiLU", tcls="d-t d-sm")
    s.path([(812, 314), (812, 298), (852, 298)])
    s.box(782, 352, 60, 24, "b-out", "Linear", tcls="d-t d-sm")
    s.path([(812, 352), (812, 338)])
    s.box(878, 352, 60, 24, "b-out", "Linear", tcls="d-t d-sm")
    s.path([(908, 352), (908, 298), (868, 298)])
    s.path([(812, 392), (908, 392)], arrow=False)
    s.path([(860, 404), (860, 392)], arrow=False)
    s.path([(812, 392), (812, 376)])
    s.path([(908, 392), (908, 376)])
    s.text(860, 426, "SwiGLU feed-forward", "d-s d-strong")
    s.save("llm-architectures.svg")


# ---------------------------------------------------------------------------
# 2. Encoder-decoder transformer (Vaswani et al., 2017)
# ---------------------------------------------------------------------------
def transformer():
    s = SVG("tf", 640, 744, "The encoder-decoder transformer architecture")
    enc, dec = 190, 450

    def embed_and_pe(cx, emb_label, src_label, side):
        s.op(cx, 648, r=9)
        s.path([(cx, 639), (cx, 544 if cx == dec else 580)])
        s.box(cx - 80, 672, 160, 28, "b-embed", emb_label)
        s.path([(cx, 672), (cx, 657)])
        s.path([(cx, 716), (cx, 700)])
        s.text(cx, 734, src_label, "d-s")
        ix = cx + side * 48
        s.path([(cx + side * 9, 648), (ix - side * 14, 648)], arrow=False)
        s.add(f'<circle cx="{ix}" cy="648" r="14" class="b-pos"/>')
        wave = " L".join(f"{ix-10+i:g},{648-6*math.sin(i/20*2*math.pi):.2f}" for i in range(21))
        s.add(f'<path d="M{wave}" class="d-line"/>')
        tx, anchor = ix + side * 22, "start" if side > 0 else "end"
        s.text(tx, 644, "Positional", "d-s", anchor)
        s.text(tx, 659, "encoding", "d-s", anchor)

    def qkv(cx, y_branch, y_top):
        s.dot(cx, y_branch)
        s.path([(cx, y_branch), (cx - 40, y_branch), (cx - 40, y_top)])
        s.path([(cx, y_branch), (cx + 40, y_branch), (cx + 40, y_top)])

    # Encoder
    s.add(f'<rect x="{enc-100}" y="386" width="200" height="230" rx="10" class="d-group"/>')
    s.text(enc - 108, 506, "N×", "d-s d-strong", "end")
    s.box(enc - 80, 400, 160, 26, "b-norm", "Add & norm")
    s.box(enc - 80, 440, 160, 28, "b-ffn", "Feed-forward")
    s.path([(enc, 440), (enc, 426)])
    s.box(enc - 80, 496, 160, 26, "b-norm", "Add & norm")
    s.path([(enc, 496), (enc, 468)])
    s.dot(enc, 482)
    s.path([(enc, 482), (enc - 92, 482), (enc - 92, 413), (enc - 80, 413)])
    s.box(enc - 80, 540, 160, 40, "b-attn", ["Multi-head", "attention"])
    s.path([(enc, 540), (enc, 522)])
    s.dot(enc, 606)
    s.path([(enc, 606), (enc - 92, 606), (enc - 92, 509), (enc - 80, 509)])
    qkv(enc, 596, 580)
    embed_and_pe(enc, "Input embedding", "Inputs", -1)

    # Decoder
    s.add(f'<rect x="{dec-100}" y="146" width="200" height="470" rx="10" class="d-group"/>')
    s.text(dec + 108, 380, "×N", "d-s d-strong", "start")
    s.text(dec, 30, "Output probabilities", "d-s")
    s.box(dec - 80, 50, 160, 28, "b-out", "Softmax")
    s.path([(dec, 50), (dec, 38)])
    s.box(dec - 80, 98, 160, 28, "b-out", "Linear")
    s.path([(dec, 98), (dec, 78)])
    s.box(dec - 80, 160, 160, 26, "b-norm", "Add & norm")
    s.path([(dec, 160), (dec, 126)])
    s.box(dec - 80, 204, 160, 30, "b-ffn", "Feed-forward")
    s.path([(dec, 204), (dec, 186)])
    s.dot(dec, 250)
    s.path([(dec, 250), (dec + 92, 250), (dec + 92, 173), (dec + 80, 173)])
    s.box(dec - 80, 266, 160, 26, "b-norm", "Add & norm")
    s.path([(dec, 266), (dec, 234)])
    s.box(dec - 80, 330, 160, 44, "b-attn", ["Multi-head", "attention"])
    s.path([(dec, 330), (dec, 292)])
    s.path([(dec, 438), (dec, 374)])
    s.dot(dec, 420)
    s.path([(dec, 420), (dec + 92, 420), (dec + 92, 279), (dec + 80, 279)])
    s.box(dec - 80, 438, 160, 26, "b-norm", "Add & norm")
    s.box(dec - 80, 500, 160, 44, "b-attn", ["Masked multi-head", "attention"])
    s.path([(dec, 500), (dec, 464)])
    s.dot(dec, 584)
    s.path([(dec, 584), (dec + 92, 584), (dec + 92, 451), (dec + 80, 451)])
    qkv(dec, 566, 544)
    embed_and_pe(dec, "Output embedding", "Outputs (shifted right)", 1)

    # Encoder output feeds keys and values of the decoder's cross-attention.
    s.path([(enc, 400), (enc, 366), (318, 366), (318, 392), (dec - 50, 392), (dec - 50, 374)])
    s.dot(dec - 28, 392)
    s.path([(dec - 28, 392), (dec - 28, 374)])
    s.save("transformer.svg")


# ---------------------------------------------------------------------------
# 3. Tokenization
# ---------------------------------------------------------------------------
TOKENS = ["The", "mind", "of", "man", "is", "capable", "of", "anything", "."]
IDS = [791, 4059, 315, 893, 374, 13171, 315, 4205, 13]


def tokenization():
    s = SVG("tok", 700, 184, "A sentence split into tokens, each mapped to an integer id")
    s.text(350, 34, "The mind of man is capable of anything.", "d-sentence")
    s.path([(350, 50), (350, 92)])
    s.text(362, 76, "tokenize", "d-s d-accent", "start")
    widths = [7.8 * len(t) + 20 for t in TOKENS]
    x = (700 - (sum(widths) + 8 * (len(TOKENS) - 1))) / 2
    s.text(x - 12, 125, "tokens", "d-s", "end")
    s.text(x - 12, 164, "ids", "d-s", "end")
    for tok, tid, w in zip(TOKENS, IDS, widths):
        s.box(round(x, 1), 104, round(w, 1), 32, "b-token", tok, tcls="d-mono")
        s.text(round(x + w / 2, 1), 164, str(tid), "d-mono d-muted")
        x += w + 8
    s.save("tokenization.svg")


# ---------------------------------------------------------------------------
# 4. Embedding lookup
# ---------------------------------------------------------------------------
def embedding():
    s = SVG("emb", 700, 330, "Token ids mapped through the embedding layer to dense vectors")
    top, step = 46, 28
    s.text(65, 26, "tokens", "d-s")
    s.text(200, 26, "ids", "d-s")
    s.text(504, 26, "embeddings  (L × d_model)", "d-s")
    for i, (tok, tid) in enumerate(zip(TOKENS, IDS)):
        y = top + i * step
        s.box(20, y, 90, 22, "b-token", tok, rx=5, tcls="d-mono d-sm")
        s.text(200, y + 15.5, str(tid), "d-mono")
        for j in range(12):
            v = math.sin(tid * 0.37 + j * 1.91) * math.cos(tid * 0.011 + j * 0.53)
            cx = 320 + j * 29 + (22 if j >= 10 else 0)
            cls = "c-pos" if v >= 0 else "c-neg"
            s.add(f'<rect x="{cx}" y="{y}" width="26" height="22" rx="3" class="{cls}" fill-opacity="{0.12 + 0.82 * abs(v):.2f}"/>')
        s.text(320 + 10 * 29 + 8, y + 15, "…", "d-s")
    mid = top + 4 * step + 11
    s.path([(118, mid), (166, mid)])
    s.text(142, mid - 10, "lookup", "d-xs")
    s.path([(234, mid), (306, mid)])
    s.text(270, mid - 22, "embedding", "d-xs")
    s.text(270, mid - 10, "layer", "d-xs")
    s.path([(320, 316), (688, 316)], arrow=False, cls="d-leader")
    s.text(504, 310, "d_model = 512", "d-xs")
    s.save("embedding.svg")


# ---------------------------------------------------------------------------
# 5. Scaled dot-product attention
# ---------------------------------------------------------------------------
def scaled_dot_product():
    s = SVG("sdpa", 540, 384, "Scaled dot-product attention")
    s.path([(190, 40), (190, 12)])
    s.box(90, 40, 200, 28, "b-attn", "MatMul")
    s.box(90, 100, 120, 28, "b-embed", "Softmax")
    s.path([(150, 100), (150, 68)])
    s.box(90, 160, 120, 28, "b-plain d-dashed", "Mask (optional)", tcls="d-t d-sm")
    s.path([(150, 160), (150, 128)])
    s.box(90, 220, 120, 28, "b-norm", "Scale")
    s.path([(150, 220), (150, 188)])
    s.box(90, 280, 120, 28, "b-attn", "MatMul")
    s.path([(150, 280), (150, 248)])
    for x, name in [(120, "Q"), (180, "K"), (260, "V")]:
        s.text(x, 370, name, "d-t d-strong")
        s.path([(x, 352), (x, 308 if name != "V" else 68)])
    notes = [
        (58, "weighted sum of the values"),
        (118, "each row becomes a distribution"),
        (178, "future positions set to −∞"),
        (238, 'divide by √d<tspan class="d-sub" dy="3">k</tspan>'),
        (298, "QKᵀ: an L × L score matrix"),
    ]
    for y, note in notes:
        s.path([(300 if y == 58 else 220, y - 4), (316, y - 4)], arrow=False, cls="d-leader")
        s.text(324, y, note, "d-s", "start", raw=True)
    s.save("scaled-dot-product-attention.svg")


# ---------------------------------------------------------------------------
# 6. Multi-head attention
# ---------------------------------------------------------------------------
def multi_head():
    s = SVG("mha", 640, 430, "Multi-head attention: h parallel scaled dot-product attention heads")

    def stacked(x, y, w, h, cls, lines):
        for k, op in ((2, "0.35"), (1, "0.6")):
            s.add(f'<rect x="{x+8*k}" y="{y-8*k}" width="{w}" height="{h}" rx="6" class="{cls}" opacity="{op}"/>')
        s.box(x, y, w, h, cls, lines)

    s.box(210, 46, 100, 28, "b-out", "Linear")
    s.path([(260, 46), (260, 16)])
    s.box(200, 106, 120, 28, "b-norm", "Concat")
    s.path([(260, 106), (260, 74)])
    stacked(130, 186, 260, 48, "b-attn", ["Scaled dot-product", "attention"])
    s.path([(260, 186), (260, 134)])
    for x, name in [(150, "V"), (260, "K"), (370, "Q")]:
        stacked(x - 45, 300, 90, 28, "b-out", "Linear")
        s.path([(x, 300), (x, 234)])
        s.path([(x, 392), (x, 328)])
        s.text(x, 412, name, "d-t d-strong")
    s.path([(414, 170), (422, 170), (422, 218), (414, 218)], arrow=False, cls="d-leader")
    notes = [
        (64, 310, 'output projection W<tspan class="d-sup" dy="-5">O</tspan>'),
        (124, 320, "join the h head outputs"),
        (198, 422, "h heads in parallel"),
        (318, 431, "per-head projections"),
    ]
    for y, x0, note in notes:
        s.path([(x0 + 6, y - 4), (446, y - 4)], arrow=False, cls="d-leader")
        s.text(454, y, note, "d-s", "start", raw=True)
    s.save("multi-head-attention.svg")


for fn in (llm_architectures, transformer, tokenization, embedding, scaled_dot_product, multi_head):
    fn()
print("ok")
