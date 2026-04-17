import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BG        = "#F2F2F2"
CARD_BG   = "#FFFFFF"
TITLE_CLR = "#141929"
BODY_CLR  = "#6B7280"
GRID_CLR  = "#E0E0E0"
BORDER    = "#D1D5DB"
BLUE      = "#4A7BDB"
PINK      = "#E84C8B"
GREEN     = "#7DC441"
ORANGE    = "#F5A623"

def style_ax(ax, xlabel, ylabel):
    ax.set_facecolor(CARD_BG)
    ax.set_xlabel(xlabel, fontsize=11, color=BODY_CLR, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color=BODY_CLR, labelpad=8)
    ax.tick_params(colors=BODY_CLR, labelsize=10, length=3)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER)
        s.set_linewidth(0.8)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.8)
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

def make_legend(ax, items):
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2.5, linestyle=ls,
                   label=lbl, marker=mk, markersize=6)
        for lbl, c, ls, mk in items
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=1,
              facecolor=CARD_BG, edgecolor=BORDER, fontsize=10,
              labelcolor=BODY_CLR, handlelength=2.6, handletextpad=0.8)


# ══════════════════════════════════════════════════════════════════════════
# CHART 1 — GA CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════
ga_gens = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ga_r1   = [0.1397, 0.0885, 0.0790, 0.0775, 0.0775, 0.0775, 0.0775, 0.0775, 0.0775, 0.0775, 0.0775]
ga_r2   = [0.1539, 0.1060, 0.1044, 0.0972, 0.0959, 0.0938, 0.0938, 0.0938, 0.0938, 0.0938, 0.0938]
ga_r3   = [0.1248, 0.1057, 0.1037, 0.1036, 0.1036, 0.1036, 0.1036, 0.0852, 0.0771, 0.0753, 0.074]

fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=BG)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)

fig.text(0.05, 0.96, "Genetic Algorithm (GA) — Energy Convergence",
         fontsize=17, fontweight="bold", color=TITLE_CLR, va="top")

ax.plot(ga_gens, ga_r1, color=BLUE,  lw=2.8, ls="-",  marker="o", ms=6, zorder=3)
ax.plot(ga_gens, ga_r2, color=PINK,  lw=2.8, ls="--", marker="s", ms=6, zorder=3)
ax.plot(ga_gens, ga_r3, color=GREEN, lw=2.8, ls=":",  marker="^", ms=7, zorder=3)

for vals, clr, dy in [(ga_r1, BLUE, 6), (ga_r2, PINK, -14), (ga_r3, GREEN, 6)]:
    ax.annotate(f"{vals[-1]:.4f}", xy=(ga_gens[-1], vals[-1]),
                xytext=(5, dy), textcoords="offset points",
                fontsize=9.5, color=clr, fontweight="bold")

style_ax(ax, "Generation", "Energy")
ax.set_xlim(-3, 110)
ax.set_ylim(0.060, 0.168)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

make_legend(ax, [
    ("Run 1: fitness 0.923", BLUE,  "-",  "o"),
    ("Run 2: fitness 0.906", PINK,  "--", "s"),
    ("Run 3: fitness 0.926", GREEN, ":",  "^"),
])

plt.savefig("report/chart_ga_convergence.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# CHART 2 — SA CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════
sa_steps = [0, 2000, 4000, 6000, 8000, 10000]
sa_r1    = [0.3476, 0.1453, 0.1453, 0.1453, 0.1453, 0.1453]
sa_r2    = [0.2492, 0.1493, 0.1331, 0.1331, 0.1331, 0.1331]
sa_r3    = [0.3451, 0.1576, 0.1576, 0.1576, 0.1576, 0.1576]

fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=BG)
fig.patch.set_facecolor(BG)
plt.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)

fig.text(0.05, 0.96, "Simulated Annealing (SA) — Energy Convergence",
         fontsize=17, fontweight="bold", color=TITLE_CLR, va="top")

ax.plot(sa_steps, sa_r1, color=BLUE,   lw=2.8, ls="-",  marker="o", ms=7, zorder=3)
ax.plot(sa_steps, sa_r2, color=PINK,   lw=2.8, ls="--", marker="s", ms=7, zorder=3)
ax.plot(sa_steps, sa_r3, color=ORANGE, lw=2.8, ls=":",  marker="^", ms=8, zorder=3)

for vals, clr, dy in [(sa_r1, BLUE, 8), (sa_r2, PINK, -15), (sa_r3, ORANGE, 8)]:
    ax.annotate(f"{vals[-1]:.4f}", xy=(sa_steps[-1], vals[-1]),
                xytext=(5, dy), textcoords="offset points",
                fontsize=9.5, color=clr, fontweight="bold")

style_ax(ax, "Step", "Energy")
ax.set_xlim(-200, 11200)
ax.set_ylim(0.09, 0.40)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

make_legend(ax, [
    ("Run 1: fitness 0.855", BLUE,   "-",  "o"),
    ("Run 2: fitness 0.867", PINK,   "--", "s"),
    ("Run 3: fitness 0.842", ORANGE, ":",  "^"),
])

plt.savefig("report/chart_sa_convergence.png", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()