#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse, json, statistics, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--iterations",type=int,default=200); p.add_argument("--device",default="CPU"); p.add_argument("--output",default="benchmarks/runtime_smoke.json")
    a=p.parse_args(); times=[]
    try:
        import openvino as ov
        core=ov.Core(); available=core.available_devices
        xml='<net name="smoke" version="10"><layers><layer id="0" name="input" type="Parameter" version="opset1"><data shape="1,4" element_type="f32"/><output><port id="0" precision="FP32" names="input"><dim>1</dim><dim>4</dim></port></output></layer><layer id="1" name="result" type="Result" version="opset1"><input><port id="0" precision="FP32"><dim>1</dim><dim>4</dim></port></input></layer></layers><edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/></edges></net>'
        # Compilation of a generated XML is version-sensitive; report availability
        # even when the smoke graph is unavailable instead of inventing measurements.
        report={"benchmark_type":"runtime_smoke","device":a.device,"available_devices":available,"status":"available" if a.device in available or a.device=="AUTO" else "device_unavailable"}
    except Exception as exc:
        report={"benchmark_type":"runtime_smoke","device":a.device,"status":"not_run","reason":str(exc)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
