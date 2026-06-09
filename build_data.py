"""CI data builder. Fetch+digitise frames and write data/<AREA>_<DAY>.json.
Run by a scheduled GitHub Action (the 'database writer'); never by the live app.
Usage: python build_data.py SSP 20260615 [SSP 20260616 ...]"""
import sys, os, json
import data as D
def main(args):
    os.makedirs("data",exist_ok=True)
    pairs=list(zip(args[0::2],args[1::2]))
    for area,day in pairs:
        print(f"building {area} {day} ...",flush=True)
        d=D.build(area,day,progress=lambda p,m:None)
        json.dump(d,open(f"data/{area}_{day}.json","w"))
        print(f"  {len(d['frames'])} frames, {len(d['missing'])} empty -> data/{area}_{day}.json")
if __name__=="__main__":
    if len(sys.argv)<3: print(__doc__); sys.exit(1)
    main(sys.argv[1:])
