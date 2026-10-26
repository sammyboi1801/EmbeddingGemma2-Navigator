"""Plot validation cross-entropy for both training stages from logs/*.log.

Writes assets/val_curves_light.svg and assets/val_curves_dark.svg (no dependencies).
usage: python scripts/plot_val_curves.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (log file, eval_every, total steps of that run, global step offset)
STAGE1 = [("soda.log", 1000, 23143, 0),       # first run, interrupted after step 7.7k
          ("soda2.log", 1000, 15443, 7000)]   # warm start from the step-7k checkpoint
STAGE2 = [("chat.log", 250, 3000, 0)]
RESUME_AT = 7000

THEMES = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  line="#2a78d6", ring="#ffffff"),
    "dark": dict(ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 line="#3987e5", ring="#0d1117"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def read_vals(runs):
    """The k-th [val] line of a run is at step k*eval_every, or at the run's last step."""
    pts = []
    for name, every, total, offset in runs:
        with open(os.path.join(ROOT, "logs", name), encoding="utf-8") as f:
            vals = [float(m[1]) for m in re.finditer(r"\[val\] ce ([0-9.]+)", f.read())]
        steps = list(range(every, total + 1, every))
        if not steps or steps[-1] != total:
            steps.append(total)
        pts += [(offset + s, v) for s, v in zip(steps, vals)]
    return pts


def nice_k(x):
    return f"{x / 1000:g}k" if x else "0"


def panel(pts, x0, y0, w, h, title, xmax, xticks, ylo, yhi, yticks, c, resume=None):
    pl, pr, pt, pb = 46, 12, 34, 34  # inner padding: y labels, right, title, x labels
    X = lambda s: x0 + pl + (w - pl - pr) * s / xmax
    Y = lambda v: y0 + pt + (h - pt - pb) * (yhi - v) / (yhi - ylo)
    out = [f'<text x="{x0 + pl}" y="{y0 + 16}" fill="{c["ink"]}" font-size="14" font-weight="600">{title}</text>']
    for v in yticks:
        out.append(f'<line x1="{x0 + pl}" x2="{x0 + w - pr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{c["grid"]}"/>')
        out.append(f'<text x="{x0 + pl - 8}" y="{Y(v) + 4:.1f}" fill="{c["muted"]}" font-size="11" text-anchor="end">{v:.1f}</text>')
    base = y0 + h - pb
    out.append(f'<line x1="{x0 + pl}" x2="{x0 + w - pr}" y1="{base}" y2="{base}" stroke="{c["axis"]}"/>')
    for s in xticks:
        out.append(f'<text x="{X(s):.1f}" y="{base + 17}" fill="{c["muted"]}" font-size="11" text-anchor="middle">{nice_k(s)}</text>')
    if resume:
        out.append(f'<line x1="{X(resume):.1f}" x2="{X(resume):.1f}" y1="{y0 + pt - 4}" y2="{base}" stroke="{c["muted"]}" stroke-dasharray="3 3"/>')
        out.append(f'<text x="{X(resume) + 6:.1f}" y="{y0 + pt + 6}" fill="{c["ink2"]}" font-size="11">resumed from 7k checkpoint</text>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(pts))
    out.append(f'<path d="{d}" fill="none" stroke="{c["line"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for s, v in pts:
        out.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="{c["line"]}" stroke="{c["ring"]}" stroke-width="2">'
                   f'<title>step {s:,}: {v:.3f}</title></circle>')
    (s0, v0), (s1, v1) = pts[0], pts[-1]  # direct labels on the first and last points only
    out.append(f'<text x="{X(s0) + 8:.1f}" y="{Y(v0) + 4:.1f}" fill="{c["ink"]}" font-size="12" font-weight="600">{v0:.2f}</text>')
    out.append(f'<text x="{X(s1):.1f}" y="{Y(v1) - 10:.1f}" fill="{c["ink"]}" font-size="12" font-weight="600" text-anchor="end">{v1:.2f}</text>')
    return out


def render(theme):
    c = THEMES[theme]
    W, H, gap = 900, 290, 36
    pw = (W - gap) / 2
    s1, s2 = read_vals(STAGE1), read_vals(STAGE2)
    body = panel(s1, 0, 0, pw, H - 26, "Stage 1: social dialogue", 23000, [0, 5000, 10000, 15000, 20000],
                 2.3, 3.6, [2.4, 2.8, 3.2, 3.6], c, resume=RESUME_AT)
    body += panel(s2, pw + gap, 0, pw, H - 26, "Stage 2: assistant fine-tune", 3000, [0, 1000, 2000, 3000],
                  3.45, 3.9, [3.5, 3.6, 3.7, 3.8, 3.9], c)
    body.append(f'<text x="{W / 2}" y="{H - 4}" fill="{c["muted"]}" font-size="12" text-anchor="middle">'
                f'Validation cross-entropy vs. training step. Stages use different validation sets.</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="{FONT}" role="img" aria-label="Validation cross-entropy curves for stage 1 and stage 2">\n'
            + "\n".join(body) + "\n</svg>\n")


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    for theme in THEMES:
        path = os.path.join(ROOT, "assets", f"val_curves_{theme}.svg")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(render(theme))
        print("wrote", os.path.relpath(path, ROOT))
