"""Fetch MPA tidal-atlas current frames for the CI data builder.
Polite by design: a fixed minimum interval between requests, an identifying User-Agent,
exponential backoff on transient errors / rate-limit / corrupt responses, and an optional
on-disk cache (set FRAME_CACHE) so a frame is fetched at most once — rebuilds reuse it and
never re-hammer MPA."""
import os, time, io, requests
from PIL import Image
BASE="https://digitalport-service.mpa.gov.sg/ta/ImageServlet"
SLOTS=[f"{h:02d}{m:02d}" for h in range(24) for m in (0,30)]
UA=("sgp-currents/1.0 (+https://github.com/cleanshiptech/sgp-currents) "
    "tidal-atlas digitiser; low-rate, runs nightly")
MIN_INTERVAL=0.5          # seconds between requests (~2 req/s max)
MAX_RETRIES=4
CACHE=os.environ.get("FRAME_CACHE")   # optional dir; frames cached here are reused
_last=[0.0]

def frame_id(area,day,hhmm): return f"TA_{area}{day}{hhmm}.gif"

def _throttle():
    wait=MIN_INTERVAL-(time.monotonic()-_last[0])
    if wait>0: time.sleep(wait)
    _last[0]=time.monotonic()

def _decodes(content):
    """A real GIF that PIL can actually open — guards against truncated/corrupt responses
    (which is what a rate-limited MPA returns: a GIF-ish header but undecodable body)."""
    if content[:3]!=b"GIF": return False
    try: Image.open(io.BytesIO(content)).verify(); return True
    except Exception: return False

def fetch_bytes(area, day, hhmm, session=None):
    """Return decodable GIF bytes, or None if MPA has no frame for that slot.
    Cached (if FRAME_CACHE set), throttled, and retried with backoff on network errors,
    429/5xx, and corrupt bodies. Raises only on the disclaimer-HTML (session/header) case."""
    fid=frame_id(area,day,hhmm); cp=os.path.join(CACHE,fid) if CACHE else None
    if cp and os.path.isfile(cp):
        return open(cp,"rb").read() or None          # cache holds only verified GIFs
    s=session or requests
    params={"id":fid}; headers={"Accept":"image/gif","User-Agent":UA}
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            r=s.get(BASE,params=params,headers=headers,timeout=30)
        except requests.RequestException:
            time.sleep(2*(attempt+1)); continue          # network hiccup -> back off
        if r.status_code in (429,500,502,503,504):
            time.sleep(min(60,5*2**attempt)); continue   # rate-limited / busy -> back off
        c=r.content
        if c[:3]!=b"GIF":
            if c[:3]==b"<!D" or c[:5]==b"<html":
                raise RuntimeError("Got disclaimer HTML, not a GIF (header/session issue).")
            return None                                  # genuinely no frame for this slot
        if not _decodes(c):
            time.sleep(min(60,5*2**attempt)); continue   # truncated/corrupt (rate-limited) -> back off
        if cp:
            os.makedirs(CACHE,exist_ok=True); open(cp,"wb").write(c)
        return c
    return None                                          # gave up after retries
