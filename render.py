"""Render speed-grid + direction-icon overlays (transparent PNG, frame pixel space),
plus date aggregates and point time-series. Glyphs are supersampled for clean edges."""
import math, io, base64
import numpy as np
from PIL import Image, ImageDraw
import extractor as ex

W,H=1673,896; CELL=17; SS=3   # half-cell px; supersample factor for anti-aliasing
# arrow glyph pointing +x: triangular head + rectangular shaft (cleaner than a flat chevron)
DART=[(10,0),(2,-6),(2,-2.4),(-9,-2.4),(-9,2.4),(2,2.4),(2,6)]
# A grid cell with no detected arrow can only be <=0.5 kn: every band >=0.5 is validated at
# ~100% recall, so by elimination it's calm (too weak to resolve a direction). Show it as the
# 0-0.5 band, no dart -- not "no reading".
CALM=ex.band_color(0.25)+(80,)

def _rot(pts,ang,cx,cy,s):
    c,si=math.cos(ang),math.sin(ang)
    return [((cx+(x*c-y*si)*s), (cy+(x*si+y*c)*s)) for x,y in pts]

def _overlay(items, cell_alpha=140, icon=(28,31,38)):
    """items: (px,py,speed,deg). speed=None -> <=0.5 kn calm cell (no dart, current too weak
    for a direction); any real speed -> band-coloured fill plus a haloed direction arrow."""
    big=Image.new("RGBA",(W*SS,H*SS),(0,0,0,0)); d=ImageDraw.Draw(big)
    cell=lambda x,y,f: d.rectangle([x-CELL*SS,y-CELL*SS,x+CELL*SS,y+CELL*SS],fill=f)
    rd=int(2.3*SS)                                 # calm-cell dot radius
    for px,py,speed,deg in items:                 # calm (<=0.5) cells: faint fill + a small dot
        if speed is None:
            x,y=px*SS,py*SS; cell(x,y,CALM)
            d.ellipse([x-rd-SS,y-rd-SS,x+rd+SS,y+rd+SS],fill=(255,255,255,200))   # halo
            d.ellipse([x-rd,y-rd,x+rd,y+rd],fill=icon+(255,))                     # dot (= dart colour)
    for px,py,speed,deg in items:                 # measured cells on top
        if speed is not None: cell(px*SS,py*SS,ex.band_color(speed)+(cell_alpha,))
    for px,py,speed,deg in items:                 # darts for every measured cell
        if speed is None: continue
        x,y=px*SS,py*SS
        ang=math.atan2(-math.cos(math.radians(deg)),math.sin(math.radians(deg)))  # heading in image coords
        d.polygon(_rot([(p[0]*1.3,p[1]*1.3) for p in DART],ang,x,y,SS),fill=(255,255,255,235))
        d.polygon(_rot(DART,ang,x,y,SS),fill=icon+(255,))
    return big.resize((W,H),Image.LANCZOS)

def frame_overlay(arrows):
    return _overlay([(a["px"],a["py"],a["speed"],a["dir"]) for a in arrows])

def gridded_frame_overlay(frames, ti, max_px=22):
    """Snapshot: draw EVERY recovered arrow at time ti (coloured cell + dart), then add a
    faint grey 'no reading' cell at any canonical-grid node with no arrow nearby — so the
    survey extent is visible while blanks stay honestly unknown (not painted 'calm').
    (Arrow positions jitter frame-to-frame, so we render the real arrows directly rather
    than snapping them onto the canonical grid, which would drop the unmatched ones.)"""
    cur=frames.get(ti,[])
    items=[(a["px"],a["py"],a["speed"],a["dir"]) for a in cur]   # all real readings
    nodes=canonical_nodes(frames)
    if cur:
        cx=np.array([a["px"] for a in cur]); cy=np.array([a["py"] for a in cur])
        for px,py,lat,lon in nodes:
            if ((cx-px)**2+(cy-py)**2).min()>max_px*max_px:
                items.append((px,py,None,0.0))   # no reading here -> grey cell
    else:
        items=[(px,py,None,0.0) for px,py,lat,lon in nodes]
    drawn=sum(1 for it in items if it[2] is not None)
    assert drawn==len(cur), f"render dropped arrows: drew {drawn} of {len(cur)} at {ti}"
    return _overlay(items)

def canonical_nodes(frames):
    dense=max(frames.values(),key=len)
    return [(a["px"],a["py"],a["lat"],a["lon"]) for a in dense]

def aggregate(frames, stat="max", spacing=35):
    half=spacing*0.6; out=[]
    for px,py,lat,lon in canonical_nodes(frames):
        sp=[]; dr=[]
        for arrows in frames.values():
            best=None
            for a in arrows:
                dd=(a["px"]-px)**2+(a["py"]-py)**2
                if best is None or dd<best[0]: best=(dd,a)
            if best and best[0]<=half*half: sp.append(best[1]["speed"]); dr.append(best[1]["dir"])
        if not sp: continue
        sp=np.array(sp)
        if stat=="max":
            i=int(sp.argmax()); val=float(sp[i]); ddeg=dr[i]
        else:
            val=float(sp.mean()); ang=np.deg2rad(dr)
            ddeg=math.degrees(math.atan2(np.sin(ang).mean(),np.cos(ang).mean()))%360
        out.append(dict(px=px,py=py,lat=lat,lon=lon,speed=val,dir=ddeg,n=len(sp)))
    return out

def aggregate_overlay(frames, stat="max"):
    return _overlay([(n["px"],n["py"],n["speed"],n["dir"]) for n in aggregate(frames,stat)])

def point_series(frames, lat, lon, max_km=1.0):
    rows=[]
    for hhmm in sorted(frames):
        r=ex.query_nearest(frames[hhmm],lat,lon,max_km=max_km)
        if r: rows.append((hhmm,r["speed"],r["dir"],r["dist_km"]))
    return rows

def png_data_uri(im, max_w=None, optimize=False):
    if max_w and im.width>max_w:
        im=im.resize((max_w, round(im.height*max_w/im.width)), Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"PNG",optimize=optimize)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

def clip_lon(im, bounds, split, keep):
    """Blank (make transparent) the half of an area overlay past `split` longitude, so SSP and
    EBA tile without overlapping -> the shared band isn't double-plotted (denser/darker). bounds
    = that area's {sw,ne}; keep='W' keeps west of split, 'E' keeps east."""
    lon0=bounds["sw"][1]; lon1=bounds["ne"][1]
    if not (min(lon0,lon1) < split < max(lon0,lon1)): return im   # split not within this overlay
    col=int(round((split-lon0)/(lon1-lon0)*im.width)); col=max(0,min(im.width,col))
    a=np.array(im.convert("RGBA"))
    if keep=="W": a[:,col:,3]=0
    else:         a[:,:col,3]=0
    return Image.fromarray(a)

def kiosk_uri(im, max_w=680, colors=32):
    """Tiny palette-PNG data URI for the TV loop: ~40KB vs ~700KB, per-pixel alpha preserved
    (basemap shows through empty cells), so hundreds of frames embed without bloating the page."""
    if im.width>max_w: im=im.resize((max_w, round(im.height*max_w/im.width)), Image.LANCZOS)
    q=im.convert("RGBA").quantize(colors=colors, method=Image.FASTOCTREE)
    b=io.BytesIO(); q.save(b,"PNG",optimize=True)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
