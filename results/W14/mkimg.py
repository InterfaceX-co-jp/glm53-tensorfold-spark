#!/usr/bin/env python3
"""W14: generate the test images (synthetic, no private content) into DIR with ground truth in DIR/truth.json.

    mkimg.py DIR        needs Pillow and the DejaVu fonts (head: Pillow 10.2, /usr/share/fonts/truetype/dejavu)
"""
import json, math, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
FD = "/usr/share/fonts/truetype/dejavu/"
def font(size, mono=False, bold=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(FD + name + ("-Bold" if bold else "") + ".ttf", size)
truth = {}

# (a) desktop screenshot 1920x1080: two windows with titles, a taskbar with a clock
im = Image.new("RGB", (1920, 1080))
d = ImageDraw.Draw(im)
for y in range(1080):
    d.line([(0, y), (1920, y)], fill=(20, 60 + y * 80 // 1080, 140 + y * 60 // 1080))
d.rectangle([0, 1040, 1920, 1080], fill=(30, 30, 36))
d.text((1810, 1050), "14:32", font=font(20), fill=(230, 230, 230))
def window(x, y, w, h, title):
    d.rectangle([x, y, x + w, y + h], fill=(250, 250, 250), outline=(90, 90, 90))
    d.rectangle([x, y, x + w, y + 36], fill=(60, 64, 72))
    d.text((x + 14, y + 7), title, font=font(20, bold=True), fill=(255, 255, 255))
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([x + w - 90 + i * 26, y + 10, x + w - 74 + i * 26, y + 26], fill=c)
window(120, 110, 900, 640, "notes.txt - Text Editor")
for i, line in enumerate(["Shopping list", "- apples", "- bread", "- coffee beans", "", "Call the dentist on Friday"]):
    d.text((145, 170 + i * 34), line, font=font(24, mono=True), fill=(20, 20, 20))
window(1150, 200, 520, 640, "Calculator")
d.rectangle([1175, 255, 1645, 345], fill=(235, 238, 240))
d.text((1560, 280), "391", font=font(40, mono=True), fill=(10, 10, 10))
keys = ["7", "8", "9", "/", "4", "5", "6", "*", "1", "2", "3", "-", "0", ".", "=", "+"]
for k, lab in enumerate(keys):
    r, c = divmod(k, 4)
    x0, y0 = 1175 + c * 118, 365 + r * 115
    d.rectangle([x0, y0, x0 + 106, y0 + 102], fill=(210, 214, 220), outline=(150, 150, 150))
    d.text((x0 + 42, y0 + 32), lab, font=font(34, bold=True), fill=(20, 20, 20))
im.save(out / "desk_1920x1080.png")
truth["desk"] = {"titles": ["notes.txt", "Calculator"], "display": "391", "clock": "14:32"}

# (a) a drawn "photo" 1024x768 JPEG: sky, sun, grass, a red house with a door, a tree
def scene(w, h):
    im = Image.new("RGB", (w, h), (135, 196, 235))
    d = ImageDraw.Draw(im)
    s = w / 1024
    d.rectangle([0, int(520 * s), w, h], fill=(76, 160, 60))
    d.ellipse([int(820 * s), int(60 * s), int(940 * s), int(180 * s)], fill=(255, 215, 40))
    d.rectangle([int(300 * s), int(330 * s), int(600 * s), int(560 * s)], fill=(190, 40, 40))
    d.polygon([(int(280 * s), int(340 * s)), (int(450 * s), int(200 * s)), (int(620 * s), int(340 * s))], fill=(90, 50, 30))
    d.rectangle([int(420 * s), int(450 * s), int(480 * s), int(560 * s)], fill=(70, 40, 20))
    d.rectangle([int(330 * s), int(380 * s), int(390 * s), int(430 * s)], fill=(200, 230, 250))
    d.rectangle([int(740 * s), int(380 * s), int(770 * s), int(560 * s)], fill=(100, 60, 30))
    d.ellipse([int(670 * s), int(260 * s), int(840 * s), int(420 * s)], fill=(30, 120, 40))
    return im
scene(1024, 768).save(out / "scene_1024x768.jpg", quality=92)
truth["scene"] = {"objects": ["house", "sun", "tree"]}

# (b) terminal screenshot 1024x768: 3 commands with their output
term = ['$ echo "hello vision"', "hello vision", '$ python3 -c "print(17*23)"', "391", "$ ls /opt/demo",
        "alpha.txt  beta.log  gamma.csv"]
im = Image.new("RGB", (1024, 768), (18, 18, 22)); d = ImageDraw.Draw(im)
d.rectangle([0, 0, 1024, 32], fill=(50, 50, 58)); d.text((12, 6), "user@demo: ~", font=font(18, bold=True), fill=(220, 220, 220))
for i, line in enumerate(term):
    d.text((20, 60 + i * 36), line, font=font(24, mono=True), fill=(120, 230, 120) if line.startswith("$") else (235, 235, 235))
im.save(out / "term_1024x768.png")
truth["term"] = "\n".join(term)

# (b) a paragraph at 12 px
para = ("The lighthouse keeper logged the weather every morning at six. On calm days the sea was grey and flat; "
        "on stormy days the waves reached the lower windows, and the lamp had to be cleaned twice.")
f12 = font(12); words = para.split(); lines, cur = [], ""
for w_ in words:
    t = (cur + " " + w_).strip()
    if d.textlength(t, font=f12) > 560: lines.append(cur); cur = w_
    else: cur = t
lines.append(cur)
im = Image.new("RGB", (600, 30 + 18 * len(lines)), (255, 255, 255)); d = ImageDraw.Draw(im)
for i, line in enumerate(lines):
    d.text((20, 15 + i * 18), line, font=f12, fill=(0, 0, 0))
im.save(out / "para_12px.png")
truth["para"] = para

# (b) a receipt
items = [("Coffee beans 500g", "12.40"), ("Oat milk", "2.15"), ("Croissant x2", "4.60"), ("Orange juice", "3.35")]
im = Image.new("RGB", (420, 520), (252, 250, 245)); d = ImageDraw.Draw(im)
d.text((120, 20), "CORNER CAFE", font=font(26, mono=True, bold=True), fill=(0, 0, 0))
d.text((110, 60), "2026-09-29  10:41", font=font(18, mono=True), fill=(40, 40, 40))
for i, (n, p) in enumerate(items):
    d.text((30, 120 + i * 40), n, font=font(20, mono=True), fill=(0, 0, 0))
    d.text((320, 120 + i * 40), p, font=font(20, mono=True), fill=(0, 0, 0))
d.line([(30, 290), (390, 290)], fill=(0, 0, 0), width=2)
d.text((30, 305), "TOTAL", font=font(22, mono=True, bold=True), fill=(0, 0, 0))
d.text((310, 305), "22.50", font=font(22, mono=True, bold=True), fill=(0, 0, 0))
im.save(out / "receipt.png")
truth["receipt"] = {"total": "22.50", "items": [n for n, _ in items]}

# (c) coloured circles on white, non-overlapping
cols = [(220, 40, 40), (40, 120, 220), (40, 170, 70), (240, 170, 20), (150, 60, 190), (20, 170, 170)]
for n in (3, 7, 12):
    rnd = random.Random(n); im = Image.new("RGB", (800, 600), (255, 255, 255)); d = ImageDraw.Draw(im); placed = []
    while len(placed) < n:
        r = rnd.randint(30, 50); x = rnd.randint(r + 10, 790 - r); y = rnd.randint(r + 10, 590 - r)
        if all(math.hypot(x - a, y - b) > r + c + 15 for a, b, c in placed):
            placed.append((x, y, r)); d.ellipse([x - r, y - r, x + r, y + r], fill=cols[len(placed) % len(cols)])
    im.save(out / f"circles_{n}.png")
    truth[f"circles_{n}"] = n

# (c) a 5 x 4 grid of star icons
im = Image.new("RGB", (700, 560), (255, 255, 255)); d = ImageDraw.Draw(im)
for r in range(4):
    for c in range(5):
        cx, cy = 80 + c * 135, 80 + r * 135
        pts = [(cx + (50 if k % 2 == 0 else 22) * math.sin(k * math.pi / 5), cy - (50 if k % 2 == 0 else 22) * math.cos(k * math.pi / 5)) for k in range(10)]
        d.polygon(pts, fill=(245, 180, 20), outline=(150, 100, 0))
im.save(out / "grid_5x4.png")
truth["grid"] = 20

# (d) two bar charts: one bar each, labelled values on the same axis
def chart(val, name, color):
    im = Image.new("RGB", (500, 400), (255, 255, 255)); d = ImageDraw.Draw(im)
    d.line([(60, 350), (460, 350)], fill=(0, 0, 0), width=2); d.line([(60, 350), (60, 30)], fill=(0, 0, 0), width=2)
    for v in range(0, 101, 20):
        y = 350 - v * 3; d.line([(55, y), (60, y)], fill=(0, 0, 0)); d.text((20, y - 8), str(v), font=font(14), fill=(0, 0, 0))
    d.rectangle([200, 350 - val * 3, 320, 350], fill=color)
    d.text((215, 360), name, font=font(18, bold=True), fill=(0, 0, 0))
    d.text((240, 350 - val * 3 - 24), str(val), font=font(18), fill=(0, 0, 0))
    return im
chart(30, "Sales", (40, 120, 220)).save(out / "chart_30.png")
chart(75, "Sales", (220, 90, 40)).save(out / "chart_75.png")
truth["charts"] = {"chart_30": 30, "chart_75": 75}

# (e) transparency: black text on a fully transparent background (reads as text on white)
im = Image.new("RGBA", (640, 200), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
d.text((40, 60), "TRANSPARENT OK 42", font=font(48, bold=True), fill=(0, 0, 0, 255))
im.save(out / "alpha_text.png")
truth["alpha"] = "TRANSPARENT OK 42"

# TTFT sizes (scene scaled; the numbers make each size's pixels unique)
for w, h in [(512, 512), (1024, 768), (1920, 1080), (3840, 2160)]:
    for v in range(3):
        im = scene(w, h); d = ImageDraw.Draw(im)
        d.text((10, 10), f"{w}x{h} #{v}", font=font(max(14, w // 60)), fill=(0, 0, 0))
        im.save(out / f"size_{w}x{h}_{v}.png")

# cache tests: two different images of the same kind
for k, (txt, bg) in enumerate([("Project ALPHA", (230, 240, 255)), ("Project OMEGA", (255, 235, 225))]):
    im = Image.new("RGB", (800, 400), bg); d = ImageDraw.Draw(im)
    d.text((60, 150), txt, font=font(60, bold=True), fill=(20, 20, 20))
    im.save(out / f"cache_{k}.png")
truth["cache"] = ["Project ALPHA", "Project OMEGA"]
json.dump(truth, open(out / "truth.json", "w"), indent=1)
print("images:", len(list(out.glob("*.png"))) + len(list(out.glob("*.jpg"))))
