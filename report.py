"""Professional PDF export for the anchorage planner (C-Leanship).
build_pdf(...) -> PDF bytes: page 1 = ETA + recommendation + 48h band heatmap; page 2 = full ranking."""
import os, io, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch
import matplotlib.image as mpimg
import extractor as ex

NAVY="#14324B"; GREEN="#1f7a4d"; GREENBG="#e8f5ee"
BANDCOLS=[tuple(c/255 for c in ex.DISPLAY_BANDS[i][1]) for i in range(6)]
LABELS=list(ex.DISPLAY_LABELS)
_LOGO=os.path.join(os.path.dirname(os.path.abspath(__file__)),"assets","logo_white.png")

def _band(v):  # kn -> 0..5 band index (matches map buckets)
    return min(5, int(v//0.5)) if v is not None else np.nan

def _header(fig, h=0.065, title="Anchorage Current Forecast", tsize=17):
    fig.add_artist(Rectangle((0,1-h),1,h,transform=fig.transFigure,color=NAVY,zorder=0))
    if os.path.isfile(_LOGO):
        im=mpimg.imread(_LOGO); ar=im.shape[1]/im.shape[0]; lw=0.20
        ax=fig.add_axes([0.035,1-h*0.82,lw,h*0.64]); ax.imshow(im); ax.axis("off")
    fig.text(0.965,1-h/2,title,ha="right",va="center",color="white",fontsize=tsize,fontweight="bold")

def _footer(fig):
    sgt=dt.datetime.utcnow()+dt.timedelta(hours=8)
    fig.text(0.035,0.022,"Source: MPA Tidal Atlas (digitised) · anchorage limits per MPA (Port) Regulations, 2nd Schedule.  "
             "Planning aid — not for navigation.",fontsize=6.5,color="#8a8a8a")
    fig.text(0.965,0.022,f"Generated {sgt:%Y-%m-%d %H:%M} SGT · c-leanship.com",fontsize=6.5,color="#8a8a8a",ha="right")

def _build_figs(eta, win, thr, am_label, rows, series, steps, nmap, approx, rec_h):
    """rows: sorted [(hrs,mean,peak,code)]; series:{code:[kn|None]}; steps:[datetime]."""
    cmap=ListedColormap(BANDCOLS); cmap.set_bad("#e2e5e8"); norm=BoundaryNorm(range(7),6)
    top=[c for h,mn,pk,c in rows if h>=rec_h]; figs=[]
    if True:
        # ---------- PAGE 1: recommendation + heatmap (landscape) ----------
        fig=plt.figure(figsize=(11.69,8.27)); fig.patch.set_facecolor("white")
        _header(fig)
        fig.text(0.035,0.885,f"Vessel ETA  {eta:%A %d %b %Y · %H:%M} SGT",fontsize=12,fontweight="bold",color=NAVY)
        fig.text(0.035,0.862,f"{win} h window   ·   calm threshold {thr:.1f} kn   ·   {am_label} current per anchorage",fontsize=9.5,color="#444")
        # recommendation banner
        fig.add_artist(Rectangle((0.035,0.775),0.93,0.066,transform=fig.transFigure,facecolor=GREENBG,edgecolor=GREEN,lw=1.4,zorder=0))
        fig.text(0.05,0.822,f"RECOMMENDED — {len(top)} anchorage{'s' if len(top)!=1 else ''} with ≥ {rec_h:.0f} h below {thr:.1f} kn",
                 fontsize=12,fontweight="bold",color=GREEN)
        rec=" ·  ".join(f"{i+1}. {nmap[c]}" for i,c in enumerate(top[:9])) if top else "None meet the criterion in this window — consider a different ETA."
        fig.text(0.05,0.792,rec,fontsize=9.5,color="#14402b")
        # heatmap
        order=[c for h,mn,pk,c in rows]
        M=np.array([[_band(v) for v in series[c]] for c in order],float)
        axh=fig.add_axes([0.235,0.085,0.62,0.64])
        axh.imshow(M,aspect="auto",cmap=cmap,norm=norm,interpolation="nearest")
        axh.set_yticks(range(len(order))); axh.set_yticklabels([(nmap[c]+(" ~" if approx.get(c) else "")) for c in order],fontsize=6.3)
        nt=len(steps); tk=[i for i in range(nt) if steps[i].minute==0 and steps[i].hour%6==0]
        axh.set_xticks(tk); axh.set_xticklabels([steps[i].strftime("%a %H:%M") for i in tk],fontsize=7,rotation=0)
        for i,c in enumerate(order):           # mark recommended rows
            if c in top: axh.add_patch(Rectangle((-0.5,i-0.5),nt,1,fill=False,edgecolor=GREEN,lw=1.1))
        axh.set_title("48-hour current by anchorage  ·  green outline = recommended",fontsize=10,loc="left",color=NAVY,pad=8)
        axh.legend(handles=[Patch(facecolor=BANDCOLS[i],label=LABELS[i]) for i in range(6)],
                   loc="upper left",bbox_to_anchor=(1.015,1.0),fontsize=7.5,title="Speed (kn)",frameon=False)
        _footer(fig); figs.append(fig)
        # ---------- PAGE 2: full ranking table (portrait) ----------
        fig=plt.figure(figsize=(8.27,11.69)); fig.patch.set_facecolor("white")
        _header(fig,h=0.05,title="Anchorage ranking",tsize=14)
        axt=fig.add_axes([0.05,0.05,0.90,0.88]); axt.axis("off")
        cols=["#","Anchorage","Code",f"Hrs <{thr:.1f}","% win","Mean kn","Peak kn"]
        data=[[str(i+1),nmap[c]+(" ~" if approx.get(c) else ""),c,f"{h:.1f}",f"{round(100*h/win)}",f"{mn:.2f}",f"{pk:.2f}"]
              for i,(h,mn,pk,c) in enumerate(rows)]
        tab=axt.table(cellText=data,colLabels=cols,loc="upper center",cellLoc="center",
                      colWidths=[0.05,0.46,0.10,0.10,0.09,0.10,0.10])
        tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1,1.3)
        for j in range(len(cols)):
            cl=tab[0,j]; cl.set_facecolor(NAVY); cl.set_text_props(color="white",fontweight="bold")
        for i,(h,mn,pk,c) in enumerate(rows):
            tab[i+1,1].get_text().set_ha("left")
            if h>=rec_h:
                for j in range(len(cols)): tab[i+1,j].set_facecolor(GREENBG)
        fig.text(0.05,0.03,f"Recommended rows (≥ {rec_h:.0f} h below {thr:.1f} kn) shaded green.  ~ = nearest-cell estimate (anchorage smaller than the grid).",
                 fontsize=7,color="#8a8a8a")
        _footer(fig); figs.append(fig)
    return figs

def build_pdf(*a, **k):
    figs=_build_figs(*a, **k); buf=io.BytesIO()
    with PdfPages(buf) as pdf:
        for f in figs: pdf.savefig(f,facecolor="white")
    for f in figs: plt.close(f)
    return buf.getvalue()
