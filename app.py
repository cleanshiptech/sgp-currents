"""MPA Current Time-Series (online). Pick a point -> current through the day.
No disk cache: reads CI-precomputed JSON (repo 'data' branch via raw URL) and falls
back to in-memory live fetch. Speed = 6-band filled grid; direction = arrows."""
import os, math, json, datetime as dt
import numpy as np, pandas as pd, streamlit as st
import altair as alt, folium
from streamlit_folium import st_folium
import extractor as ex, render as R, data as D

st.set_page_config(page_title="MPA Current Time-Series", layout="wide")
STATIONS={"SSP-A":(1.1963,103.6815),"SSP-B":(1.1893,103.6934),"SSP-C":(1.1818,103.7032),
          "SSP-D":(1.1715,103.7134),"SSP-ADCP":(1.1571,103.7402),
          "EBA-A":(1.2870,104.0020),"EBA-B":(1.3000,104.0680)}
# Greyscale Positron first: neutral basemap so the speed colours read true.
TILES={"Carto Positron":("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO"),
       "Carto Voyager":("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO"),
       "Esri Ocean":("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}","Esri — Oceans (no tiles past z13)")}

@st.cache_data(show_spinner=False, ttl=3600)   # re-check repo hourly for fresh CI data
def get_precomputed(area,day,remote): return D.load_precomputed(area,day,remote_base=remote)

def nearest_station(lat,lon):
    best=None
    for n,(la,lo) in STATIONS.items():
        d=math.hypot((la-lat)*110.6,(lo-lon)*111.3*math.cos(math.radians(lat)))
        if best is None or d<best[0]: best=(d,n)
    return best

# ---- TV kiosk mode: ?kiosk=1 -> chrome-free, auto-looping today→+6-day animation ----
_KIOSK_HTML = """<!doctype html><html><head>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{margin:0;height:100%;background:#eef1f3;font-family:system-ui,Arial,sans-serif;overflow:hidden}
 #top{position:absolute;top:0;left:0;right:0;height:58px;z-index:1000;display:flex;align-items:center;
      gap:18px;padding:0 18px;background:rgba(22,50,79,.95);color:#fff;box-sizing:border-box}
 #clock{font-size:30px;font-weight:700;white-space:nowrap}
 #legend{flex:1;text-align:center;font-size:17px}
 #legend b{margin-right:8px}
 #legend span{padding:3px 10px;margin:2px;border-radius:4px;font-weight:600;display:inline-block}
 #title{font-size:16px;opacity:.85;white-space:nowrap}
 #map{position:absolute;top:58px;left:0;right:0;bottom:0}
 #bar{position:absolute;bottom:0;left:0;height:5px;background:#2e8b57;z-index:1001;transition:width .25s linear}
</style></head><body>
<div id="top">
 <span id="clock">--</span>
 <span id="legend"><b>Speed (kn):</b> __LEGEND__</span>
 <span id="title">SGP Currents &middot; live 7-day loop</span>
</div>
<div id="map"></div>
<div id="bar"></div>
<script>
var D=__PAYLOAD__;
var BLANK='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==';
var map=L.map('map',{zoomControl:false,attributionControl:false,dragging:false,scrollWheelZoom:false,doubleClickZoom:false,boxZoom:false,keyboard:false,touchZoom:false,tap:false});
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{maxZoom:18}).addTo(map);
var ssp=D.sspb?L.imageOverlay(BLANK,[D.sspb.sw,D.sspb.ne],{opacity:.92}).addTo(map):null;
var eba=D.ebab?L.imageOverlay(BLANK,[D.ebab.sw,D.ebab.ne],{opacity:.92}).addTo(map):null;
function fit(){map.invalidateSize();map.fitBounds([D.sw,D.ne],{padding:[28,28]});}
fit();setTimeout(fit,400);setTimeout(fit,1200);window.addEventListener('resize',fit);
var i=0,n=D.frames.length;
function show(){var f=D.frames[i];
 if(ssp)ssp.setUrl(f.ssp||BLANK);
 if(eba)eba.setUrl(f.eba||BLANK);
 document.getElementById('clock').textContent=f.label;
 document.getElementById('bar').style.width=((i+1)/n*100)+'%';
 i=(i+1)%n;}
show();setInterval(show,Math.round(1000/D.fps));
setTimeout(function(){location.reload();},3*3600*1000);
</script></body></html>"""

@st.cache_data(show_spinner="Building 7-day loop…", ttl=10800)   # cache 3h; JS reloads to roll the window
def _kiosk_frames(days_tuple, step, remote):
    out=[]; sspb=ebab=None
    for ds in days_tuple:
        dssp=get_precomputed("SSP",ds,remote); deba=get_precomputed("EBA",ds,remote)
        hs=(dssp or {}).get("frames",{}); he=(deba or {}).get("frames",{})
        if not hs and not he: continue
        if dssp: sspb=dssp["bounds"]
        if deba: ebab=deba["bounds"]
        for ti in sorted(set(hs)|set(he))[::step]:
            su=R.kiosk_uri(R.gridded_frame_overlay(hs,ti)) if ti in hs else None
            eu=R.kiosk_uri(R.gridded_frame_overlay(he,ti)) if ti in he else None
            if su or eu:
                out.append({"label":f"{dt.datetime.strptime(ds,'%Y%m%d'):%a %d %b}  {ti[:2]}:{ti[2:]}","ssp":su,"eba":eu})
    return out,sspb,ebab

def _render_kiosk(qp):
    from streamlit.components.v1 import html as _html
    days=int(qp.get("days",7)); step=int(qp.get("step",2)); fps=float(qp.get("fps",2)); H=int(qp.get("h",1040))
    remote=st.secrets.get("DATA_BASE","") if hasattr(st,"secrets") else ""
    today=(dt.datetime.utcnow()+dt.timedelta(hours=8)).date()
    days_tuple=tuple((today+dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(days))
    st.markdown("<style>#MainMenu,header,footer,[data-testid='stToolbar']{display:none!important}"
                ".block-container{padding:0!important;max-width:100%!important}"
                ".stApp{overflow:hidden}"
                "iframe{height:100vh!important;width:100%!important;display:block;border:0}</style>",
                unsafe_allow_html=True)   # force the map iframe to fill the actual TV viewport
    frames,sspb,ebab=_kiosk_frames(days_tuple,step,remote)
    if not frames: st.error("No data archived yet for the next 7 days."); return
    sws=[b["sw"] for b in (sspb,ebab) if b]; nes=[b["ne"] for b in (sspb,ebab) if b]
    sw=[min(s[0] for s in sws),min(s[1] for s in sws)]; ne=[max(n[0] for n in nes),max(n[1] for n in nes)]
    def _txt(rgb): return "#111" if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2])>150 else "#fff"
    legend="".join(f"<span style='background:rgb{ex.DISPLAY_BANDS[i][1]};color:{_txt(ex.DISPLAY_BANDS[i][1])}'>{ex.DISPLAY_LABELS[i]}</span>" for i in range(6))
    payload=json.dumps({"frames":frames,"sspb":sspb,"ebab":ebab,"sw":sw,"ne":ne,"fps":fps})
    _html(_KIOSK_HTML.replace("__PAYLOAD__",payload).replace("__LEGEND__",legend),height=H,scrolling=False)

def _qp():
    try: return {k:(v[-1] if isinstance(v,list) else v) for k,v in dict(st.query_params).items()}
    except Exception: return {k:(v[0] if isinstance(v,list) else v) for k,v in st.experimental_get_query_params().items()}

if str(_qp().get("kiosk",""))=="1":
    _render_kiosk(_qp()); st.stop()

with st.sidebar:
    st.header("Data")
    area="SSP + EBA"; sel_areas=["SSP","EBA"]   # combined view only
    sgt_today=(dt.datetime.utcnow()+dt.timedelta(hours=8)).date()
    date=st.date_input("Date",sgt_today,
                       min_value=sgt_today-dt.timedelta(days=40),
                       max_value=sgt_today+dt.timedelta(days=62))
    day=date.strftime("%Y%m%d")
    remote=st.secrets.get("DATA_BASE","") if hasattr(st,"secrets") else ""    # config, not UI
    xlsx=st.secrets.get("MPA_XLSX","") if hasattr(st,"secrets") else ""
    loaded=[(a,dd) for a in sel_areas if (dd:=get_precomputed(a,day,remote)) and dd.get("frames")]
    if loaded:
        st.caption(f"✓ {date:%a %d %b %Y} · SSP + EBA")
    else:
        st.warning(f"No data for **{date:%d %b %Y}** yet. The archive covers the current month "
                   f"(it updates automatically) — try a date in {sgt_today:%B}.")
    if st.button("↻ Refresh"): get_precomputed.clear(); st.rerun()
    st.markdown("---")
    basemap=st.selectbox("Basemap",list(TILES))
    view=st.radio("Map layer",["Snapshot (time)","Aggregate: max","Aggregate: mean"])
    opacity=st.slider("Overlay opacity",0.3,1.0,0.85)

if not loaded:
    st.title("MPA Current Time-Series")
    st.info(f"No archived data for {date:%d %b %Y}. Pick a date in the current month — the "
            "archive fills automatically and extends into next month once MPA publishes it."); st.stop()

# combined bounds + union of frame times across the selected area(s)
allsw=[dd["bounds"]["sw"] for _,dd in loaded]; allne=[dd["bounds"]["ne"] for _,dd in loaded]
sw=[min(s[0] for s in allsw),min(s[1] for s in allsw)]; ne=[max(n[0] for n in allne),max(n[1] for n in allne)]
times=sorted(set().union(*[set(dd["frames"]) for _,dd in loaded]))
tcount=lambda ti: sum(len(dd["frames"].get(ti,[])) for _,dd in loaded)     # arrows across all areas
daymax=max((tcount(t) for t in times),default=0)

hcol,bcol=st.columns([5,1])
hcol.subheader(f"{area} · {date:%d %b %Y}")
hcol.caption("precomputed · click the map to drop a worksite")
if "pt" in st.session_state and bcol.button("Clear pin"): del st.session_state["pt"]; st.rerun()

left,right=st.columns([3,2])
with left:
    if view=="Snapshot (time)":
        ti=st.select_slider("Time (SGT)",times,value=times[len(times)//2])
        overlays=[(R.gridded_frame_overlay(dd["frames"],ti),dd["bounds"]) for _,dd in loaded if ti in dd["frames"]]
        cap=f"{area} {day} {ti} · {tcount(ti)} arrows"
        if daymax and tcount(ti)<0.25*daymax:
            st.info("⚓ **Slack water** — currents near zero, so most cells are calm (≤0.5 kn) with no direction. Slide to a flood/ebb time for the working current.")
    else:
        stat="max" if "max" in view else "mean"
        overlays=[(R.aggregate_overlay(dd["frames"],stat),dd["bounds"]) for _,dd in loaded]
        cap=f"{area} {day} · {stat} over day"
    url,attr=TILES[basemap]
    m=folium.Map(location=[(sw[0]+ne[0])/2,(sw[1]+ne[1])/2],zoom_start=11 if len(loaded)>1 else 12,tiles=url,attr=attr)
    for ov,bnd in overlays:
        folium.raster_layers.ImageOverlay(R.png_data_uri(ov),bounds=[bnd["sw"],bnd["ne"]],opacity=opacity).add_to(m)
    if len(loaded)>1: m.fit_bounds([sw,ne])
    if "pt" in st.session_state: folium.Marker(st.session_state["pt"]).add_to(m)
    st.caption(cap)
    ev=st_folium(m,height=470,use_container_width=True,returned_objects=["last_clicked"])
    if ev and ev.get("last_clicked"): st.session_state["pt"]=[ev["last_clicked"]["lat"],ev["last_clicked"]["lng"]]
    # 6-band legend (dark text on light swatches, light on dark — stays readable)
    def _txt(rgb): return "#111" if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2])>150 else "#fff"
    def _chip(rgb,lab): return f"<span style='background:rgb{rgb};padding:2px 8px;margin:2px;color:{_txt(rgb)};border:1px solid #cbd2d9;border-radius:3px'>{lab}</span>"
    chips="".join(_chip(ex.DISPLAY_BANDS[i][1],ex.DISPLAY_LABELS[i]) for i in range(6))
    st.markdown("**Speed (kn):** "+chips,unsafe_allow_html=True)
    st.caption("Cells without an arrow are **≤0.5 kn (calm)** — too weak to resolve a direction. "
               "All currents ≥0.5 kn are detected, so a calm cell isn't hiding stronger flow.")

with right:
    st.subheader("Current at point")
    if "pt" not in st.session_state: st.info("Click the map to choose a worksite."); st.stop()
    plat,plon=st.session_state["pt"]; st.caption(f"{plat:.4f} N, {plon:.4f} E")
    # pick whichever loaded area has the nearest arrow to the point
    best=None
    for a,dd in loaded:
        s=R.point_series(dd["frames"],plat,plon,max_km=1.0)
        if s:
            mind=min(r[3] for r in s)
            if best is None or mind<best[0]: best=(mind,a,s)
    if best is None: st.warning("No arrows within 1 km (land or outside field)."); st.stop()
    _,parea,ser=best
    if len(loaded)>1: st.caption(f"area: {parea}")
    df=pd.DataFrame(ser,columns=["time","speed","dir","dist_km"])
    df["t"]=pd.to_datetime(df["time"],format="%H%M").dt.strftime("%H:%M")
    c1,c2,c3=st.columns(3)
    c1.metric("Max (binned)",f"{df.speed.max():.1f} kn"); c2.metric("Mean",f"{df.speed.mean():.1f} kn")
    c3.metric(f"% ≥ {ex.ROV_LIMIT}",f"{100*(df.speed>=ex.ROV_LIMIT).mean():.0f}%")
    bandrows=pd.DataFrame({"y0":[0,.5,1,1.5,2,2.5],"y1":[.5,1,1.5,2,2.5,3.2],
        "c":[f"rgb{ex.DISPLAY_BANDS[i][1]}" for i in range(6)]})
    bg=alt.Chart(bandrows).mark_rect(opacity=.18).encode(y="y0:Q",y2="y1:Q",
        color=alt.Color("c:N",scale=None,legend=None))
    base=alt.Chart(df).encode(x=alt.X("t:N",title="SGT"))
    err=base.mark_area(opacity=.25,color="#333").encode(y=alt.Y("lo:Q",title="speed (kn)"),y2="hi:Q") \
        .transform_calculate(lo="datum.speed-0.25",hi="datum.speed+0.25")
    line=base.mark_line(point=True,color="#111").encode(y="speed:Q")
    limit=alt.Chart(pd.DataFrame({"y":[ex.ROV_LIMIT]})).mark_rule(strokeDash=[5,4],color="red").encode(y="y:Q")
    st.altair_chart((bg+err+line+limit).properties(height=240),use_container_width=True)
    st.altair_chart(alt.Chart(df).mark_line(point=True,color="#555").encode(
        x=alt.X("t:N",title="SGT"),y=alt.Y("dir:Q",title="set °T",scale=alt.Scale(domain=[0,360]))
        ).properties(height=140),use_container_width=True)
    dkm,stn=nearest_station(plat,plon)
    if dkm<0.4 and os.path.isfile(xlsx):
        try:
            xa=pd.read_excel(xlsx,sheet_name="Current Data"); d=dt.datetime.strptime(day,"%Y%m%d")
            es=xa[(xa.Area==parea)&(xa.Month==d.month)&(xa.Day==d.day)&(xa.Station==stn.split("-")[1])]
            if not es.empty:
                es=es.assign(t=es.Hour_SGT.map(lambda h:f"{int(h):02d}:00"))
                st.caption(f"dashed = exact MPA value at {stn} ({dkm*1000:.0f} m)")
                st.altair_chart((line+alt.Chart(es).mark_line(strokeDash=[4,3],color="black").encode(
                    x="t:N",y="Speed_knots:Q")).properties(height=200),use_container_width=True)
        except Exception: pass
    st.caption("Speed colour/bin-derived (±0.25 kn band). Red dashed = ROV limit.")
