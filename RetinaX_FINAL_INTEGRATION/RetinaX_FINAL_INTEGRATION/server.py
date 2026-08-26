from flask import Flask,request,jsonify,send_from_directory
from pathlib import Path
import subprocess,uuid,json,os

BASE=Path(__file__).resolve().parent; FRONTEND=BASE/"frontend"; RUNTIME=BASE/"runtime"; UP=RUNTIME/"uploads"; OUT=RUNTIME/"outputs"; MATLAB=BASE/"matlab"
UP.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
app=Flask(__name__,static_folder=None)
MATLAB_EXE=os.environ.get("MATLAB_EXE","matlab")

@app.get("/")
def home(): return send_from_directory(FRONTEND,"index.html")
@app.get("/<path:path>")
def static_file(path):
 p=FRONTEND/path
 return send_from_directory(FRONTEND,path) if p.exists() else ("Not found",404)
@app.get("/outputs/<case>/<path:name>")
def output(case,name): return send_from_directory(OUT/case,name)

@app.post("/api/analyze")
def analyze():
 if "image" not in request.files:return jsonify(error="No image uploaded"),400
 f=request.files["image"]; case=uuid.uuid4().hex[:12]; od=OUT/case; od.mkdir()
 suffix=Path(f.filename).suffix.lower() or ".png"; ip=UP/f"{case}{suffix}"; f.save(ip)
 q=lambda p:str(p).replace("\\","/").replace("'","''")
 cmd=f"addpath('{q(MATLAB)}');trust_dr_web_integrated('{q(ip)}','{q(od)}');"
 try:p=subprocess.run([MATLAB_EXE,"-batch",cmd],capture_output=True,text=True,timeout=1200)
 except FileNotFoundError:return jsonify(error="MATLAB executable not found. Set MATLAB_EXE or add MATLAB to PATH."),500
 except subprocess.TimeoutExpired:return jsonify(error="MATLAB inference timed out."),504
 if p.returncode != 0:

    print("\n================ MATLAB STDOUT ================\n")
    print(p.stdout)

    print("\n================ MATLAB STDERR ================\n")
    print(p.stderr)

    print("\n================================================\n")

    return jsonify(
        error="MATLAB pipeline failed.",
        matlab_stdout=p.stdout[-7000:],
        matlab_stderr=p.stderr[-7000:]
    ), 500
 rp=od/"result.json"
 if not rp.exists():return jsonify(error="MATLAB completed but result.json was not created.",matlab_stdout=p.stdout[-7000:]),500
 d=json.loads(rp.read_text(encoding="utf-8"));d["case_id"]=case
 for k in ["original","clahe","vessel_overlay","lesion_overlay","combined_overlay","gradcam_overlay","gradcam_heatmap"]:
  fn=d.get(k+"_file")
  if fn:d[k+"_url"]=f"/outputs/{case}/{fn}"
 return jsonify(d)

@app.post("/api/review/<case>")
def review(case):
 od=OUT/case
 if not od.exists():return jsonify(error="Unknown case"),404
 body=request.get_json(silent=True) or {}
 review={"decision":body.get("decision","PENDING_HUMAN_REVIEW"),"note":body.get("note","")}
 (od/"human_review.json").write_text(json.dumps(review,indent=2),encoding="utf-8")
 return jsonify(ok=True,**review)

if __name__=="__main__":
 print("RetinaX bridge: http://127.0.0.1:5000")
 app.run(host="127.0.0.1",port=5000,debug=False)
