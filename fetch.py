"""Fetch MPA tidal-atlas current frames for the CI data builder.
Polite by design: a fixed minimum interval between requests, an identifying User-Agent,
and exponential backoff on transient errors / rate-limit responses, so a full-month
build doesn't hammer the MPA service or get the runner blocked."""
import time, requests
BASE="https://digitalport-service.mpa.gov.sg/ta/ImageServlet"
SLOTS=[f"{h:02d}{m:02d}" for h in range(24) for m in (0,30)]
UA=("sgp-currents/1.0 (+https://github.com/cleanshiptech/sgp-currents) "
    "tidal-atlas digitiser; low-rate, runs nightly")
MIN_INTERVAL=0.5          # seconds between requests (~2 req/s max)
MAX_RETRIES=4
_last=[0.0]               # monotonic timestamp of the previous request (module-wide)

def frame_id(area,day,hhmm): return f"TA_{area}{day}{hhmm}.gif"

def _throttle():
    wait=MIN_INTERVAL-(time.monotonic()-_last[0])
    if wait>0: time.sleep(wait)
    _last[0]=time.monotonic()

def fetch_bytes(area, day, hhmm, session=None):
    """Return GIF bytes, or None if MPA has no frame for that slot.
    Throttled + retried; raises only on the disclaimer-HTML (session/header) case."""
    s=session or requests
    params={"id":frame_id(area,day,hhmm)}; headers={"Accept":"image/gif","User-Agent":UA}
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            r=s.get(BASE,params=params,headers=headers,timeout=30)
        except requests.RequestException:
            time.sleep(2*(attempt+1)); continue          # network hiccup -> back off
        if r.status_code in (429,500,502,503,504):
            time.sleep(min(60,5*2**attempt)); continue    # rate-limited / server busy -> back off
        if r.content[:3]!=b"GIF":
            if r.content[:3]==b"<!D" or r.content[:5]==b"<html":
                raise RuntimeError("Got disclaimer HTML, not a GIF (header/session issue).")
            return None                                   # genuinely no frame for this slot
        return r.content
    return None                                           # gave up after retries
