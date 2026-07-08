import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os, time, threading, json

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from data_loader import load_all
    from solver      import run_solver
    from visualizer  import plot_shelf_layout, plot_utilization, plot_category_balance
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

st.set_page_config(page_title = "MILP Shelf Allocator", layout = "wide", initial_sidebar_state = "expanded")

if not IMPORTS_OK:
    st.error(f"Startup error: {IMPORT_ERROR}")
    st.stop()

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  .top-bar { display:flex; align-items:center; gap:14px; padding:1.4rem 0 0.4rem 0; }
  .top-bar-icon { width:42px; height:42px; background:#1E3A5F; border-radius:10px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
  .top-bar-title { font-size:22px; font-weight:700; color:#0F1F33; margin:0; }
  .top-bar-sub   { font-size:13px; color:#6B7280; margin:0; }
  .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:1.2rem 0; }
  .kpi-card { background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:1.1rem 1.3rem; box-shadow:0 1px 3px rgba(0,0,0,.06); }
  .kpi-icon-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
  .kpi-icon { width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; }
  .kpi-tag  { font-size:11px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; color:#6B7280; }
  .kpi-val  { font-size:28px; font-weight:700; color:#0F1F33; line-height:1.1; }
  .kpi-sub  { font-size:12px; color:#9CA3AF; margin-top:4px; }
  .kpi-badge-pos { font-size:13px; font-weight:600; color:#059669; }
  .kpi-badge-neg { font-size:13px; font-weight:600; color:#DC2626; }
  .sec-head { display:flex; align-items:center; gap:9px; margin:1.8rem 0 0.9rem 0; padding-bottom:8px; border-bottom:1.5px solid #E5E7EB; }
  .sec-head-icon { width:28px; height:28px; border-radius:6px; background:#F3F4F6; display:flex; align-items:center; justify-content:center; }
  .sec-head-text { font-size:16px; font-weight:600; color:#111827; }
  .status-ok  { display:inline-block; background:#D1FAE5; color:#065F46; font-size:12px; font-weight:600; padding:3px 10px; border-radius:20px; margin-left:10px; }
  .status-err { display:inline-block; background:#FEE2E2; color:#991B1B; font-size:12px; font-weight:600; padding:3px 10px; border-radius:20px; margin-left:10px; }
  .progress-wrap { background:#F3F4F6; border-radius:12px; padding:1.4rem 1.6rem; margin:1rem 0; }
  .progress-title { font-size:14px; font-weight:600; color:#111827; margin-bottom:4px; }
  .progress-sub   { font-size:12px; color:#6B7280; margin-bottom:12px; }
  section[data-testid="stSidebar"] { background:#0F1F33 !important; }
  section[data-testid="stSidebar"] * { color:#E5E7EB !important; }
</style>
""", unsafe_allow_html=True)

def icon(d, color="#1E3A5F", size=18):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="{d}"/></svg>')

ICONS = {
    "layers"  : "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
    "trending": "M23 6l-9.5 9.5-5-5L1 18",
    "box"     : "M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z",
    "check"   : "M22 11.08V12a10 10 0 11-5.93-9.14M22 4L12 14.01l-3-3",
    "bar"     : "M18 20V10M12 20V4M6 20v-6",
    "grid"    : "M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z",
    "download": "M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3",
    "table"   : "M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18",
    "cube"    : "M12 2l10 6v8l-10 6L2 16V8l10-6z M12 22V12 M22 8l-10 4L2 8",
    "zap"     : "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
    "info"    : "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 8v4M12 16h.01"
}

def sec_header(icon_key, title, badge=None):
    badge_html = f'<span class="{"status-ok" if badge[1] else "status-err"}">{badge[0]}</span>' if badge else ""
    st.markdown(f'<div class="sec-head"><div class="sec-head-icon">{icon(ICONS[icon_key],"#374151",15)}</div><span class="sec-head-text">{title}</span>{badge_html}</div>', unsafe_allow_html=True)

@st.cache_data
def get_data():
    try: return load_all()
    except Exception as e:
        st.error(f"Could not load Excel file: {e}")
        st.stop()

data = get_data()

CATEGORY_COLORS_HEX = {"Tops": "#4CAF50", "Denim": "#2196F3", "Shirts": "#FF9800", "Dresses": "#E91E63", "Bottoms": "#9C27B0", "Ethnic": "#FF5722", "Outerwear": "#607D8B"}

def render_3d_viewer(layout_df: pd.DataFrame = None, height: int = 500):
    if layout_df is not None and not layout_df.empty:
        products_js = []
        for _, row in layout_df.iterrows():
            loc = str(row["Location_ID"])
            try:
                parts   = loc.split("R")
                s_level = int(parts[0].replace("S",""))
                r_num   = int(parts[1])
            except:
                s_level, r_num = 1, 1
            products_js.append({
                "name"    : str(row["Product_Name"]),
                "cat"     : str(row["Category"]),
                "mode"    : str(row["Display_Mode"]),
                "facings" : int(row["Facings"]),
                "rack"    : r_num,
                "level"   : s_level,
                "color"   : CATEGORY_COLORS_HEX.get(str(row["Category"]), "#607D8B"),
            })
        products_json = json.dumps(products_js)
        title_text = "Optimized Layout — 3D View"
    else:
        products_json = "[]"
        title_text = "3D Shelf Viewer — Run optimizer to see layout"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d0d1a;font-family:Inter,sans-serif;overflow:hidden}}
#wrap{{width:100%;height:{height}px;position:relative}}
canvas{{width:100%;height:100%;display:block}}
#controls{{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap}}
.btn{{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
  color:#fff;font-size:11px;padding:5px 11px;border-radius:6px;cursor:pointer;
  backdrop-filter:blur(4px);transition:background .2s}}
.btn:hover{{background:rgba(255,255,255,.2)}}
.btn.active{{background:rgba(96,165,250,.3);border-color:#60A5FA}}
#legend{{position:absolute;bottom:10px;left:10px;display:flex;flex-wrap:wrap;gap:6px}}
.leg{{display:flex;align-items:center;gap:5px;background:rgba(0,0,0,.45);
  padding:4px 8px;border-radius:5px;font-size:10px;color:#ddd}}
.leg-dot{{width:10px;height:10px;border-radius:3px;flex-shrink:0}}
#tooltip{{position:absolute;top:10px;right:10px;background:rgba(10,15,35,.9);
  color:#e5e7eb;font-size:12px;padding:10px 14px;border-radius:8px;
  pointer-events:none;line-height:1.7;min-width:160px;display:none;
  border:1px solid rgba(96,165,250,.3)}}
#title{{position:absolute;top:10px;left:50%;transform:translateX(-50%);
  background:rgba(0,0,0,.5);color:#9CA3AF;font-size:11px;padding:4px 12px;
  border-radius:20px;white-space:nowrap}}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="controls">
    <button class="btn active" onclick="setView('iso')">Isometric</button>
    <button class="btn" onclick="setView('front')">Front</button>
    <button class="btn" onclick="setView('side')">Side</button>
    <button class="btn" onclick="setView('top')">Top</button>
    <button class="btn" id="rot-btn" onclick="toggleRotate()">Auto Rotate</button>
  </div>
  <div id="title">{title_text}</div>
  <div id="legend">
    <div class="leg"><div class="leg-dot" style="background:#4CAF50"></div>Tops</div>
    <div class="leg"><div class="leg-dot" style="background:#2196F3"></div>Denim</div>
    <div class="leg"><div class="leg-dot" style="background:#FF9800"></div>Shirts</div>
    <div class="leg"><div class="leg-dot" style="background:#E91E63"></div>Dresses</div>
    <div class="leg"><div class="leg-dot" style="background:#9C27B0"></div>Bottoms</div>
    <div class="leg"><div class="leg-dot" style="background:#FF5722"></div>Ethnic</div>
    <div class="leg"><div class="leg-dot" style="background:#607D8B"></div>Outerwear</div>
  </div>
  <div id="tooltip"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const PRODUCTS = {products_json};
const RACK_W=4, RACK_GAP=1.2, N_RACKS=5;
const shelfYs = [0.5, 2.7, 4.9];
const levelToY = {{1:shelfYs[2], 2:shelfYs[1], 3:shelfYs[0]}};
const rackXs = Array.from({{length:N_RACKS}},(_,i)=>i*(RACK_W+RACK_GAP));

const canvas = document.getElementById('c');
const wrap   = document.getElementById('wrap');
const tt     = document.getElementById('tooltip');
const renderer = new THREE.WebGLRenderer({{canvas,antialias:true,alpha:false}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled = true;
renderer.setClearColor(0x0d0d1a);

const scene  = new THREE.Scene();
scene.fog    = new THREE.Fog(0x0d0d1a,35,80);
const camera = new THREE.PerspectiveCamera(42,1,0.1,200);
const CENTER = new THREE.Vector3(10,4,0);

function resize(){{
  const w=wrap.clientWidth,h=wrap.clientHeight;
  renderer.setSize(w,h);
  camera.aspect=w/h;
  camera.updateProjectionMatrix();
}}
resize();
window.addEventListener('resize',resize);

scene.add(new THREE.AmbientLight(0xffffff,0.5));
const dir=new THREE.DirectionalLight(0xffffff,0.9);
dir.position.set(10,20,15);
dir.castShadow=true;
scene.add(dir);
const fill=new THREE.PointLight(0x4488ff,0.4,60);
fill.position.set(-5,12,10);
scene.add(fill);

const floor=new THREE.Mesh(new THREE.PlaneGeometry(60,40), new THREE.MeshLambertMaterial({{color:0x0a0a18}}));
floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor);
scene.add(new THREE.GridHelper(40,20,0x222244,0x111133));

rackXs.forEach(rx=>{{
  [-RACK_W/2+0.05,RACK_W/2-0.05].forEach(xo=>{{
    const p=new THREE.Mesh(new THREE.BoxGeometry(0.08,7,0.08), new THREE.MeshLambertMaterial({{color:0x1a1a3a}}));
    p.position.set(rx+xo,3.5,0); scene.add(p);
  }});
  shelfYs.forEach(sy=>{{
    const board=new THREE.Mesh(new THREE.BoxGeometry(RACK_W-0.2,0.07,1.8), new THREE.MeshLambertMaterial({{color:0x2a2a4a}}));
    board.position.set(rx,sy,0); board.receiveShadow=true; scene.add(board);
    board.add(new THREE.LineSegments(new THREE.EdgesGeometry(board.geometry), new THREE.LineBasicMaterial({{color:0x4444aa}})));
  }});
}});

const meshObjects=[];
const slots={{}};
PRODUCTS.forEach(p=>{{
  const key=`${{p.rack}}-${{p.level}}`;
  if(!slots[key]) slots[key]=[];
  slots[key].push(p);
}});

Object.entries(slots).forEach(([key,prods])=>{{
  const [rStr,lStr]=key.split('-');
  const rx=rackXs[(parseInt(rStr)-1)];
  const sy=levelToY[parseInt(lStr)];
  const n=prods.length;
  const slotW=(RACK_W-0.3)/Math.max(n,1);

  prods.forEach((p,i)=>{{
    const xOff=-RACK_W/2+0.15+slotW*(i+0.5);
    const isHang=p.mode==='Hanging';
    const col=parseInt(p.color.replace('#',''),16);

    let geo,yOff,zOff;
    if(isHang){{
      geo=new THREE.BoxGeometry(slotW*0.72,1.1,0.12);
      yOff=0.6; zOff=0;
    }} else {{
      const stackH=0.2*p.facings;
      geo=new THREE.BoxGeometry(slotW*0.8,stackH,0.55);
      yOff=0.05+stackH/2; zOff=-0.3;
    }}

    const mesh=new THREE.Mesh(geo,new THREE.MeshLambertMaterial({{color:col}}));
    mesh.position.set(rx+xOff,sy+yOff,zOff); mesh.castShadow=true; mesh.userData={{product:p}};
    scene.add(mesh); meshObjects.push(mesh);

    if(isHang){{
      const hanger=new THREE.Mesh(new THREE.TorusGeometry(0.07,0.013,8,16,Math.PI), new THREE.MeshLambertMaterial({{color:0xcccccc}}));
      hanger.position.set(rx+xOff,sy+1.2,zOff); hanger.rotation.z=Math.PI; scene.add(hanger);
    }}
  }});
}});

let camR=22,camTheta=0.55,camPhi=0.32,rotating=false;
let isDrag=false,lastX=0,lastY=0,hoveredMesh=null,origColor=null;

function updateCam(){{
  camera.position.set(CENTER.x+camR*Math.sin(camTheta)*Math.cos(camPhi), CENTER.y+camR*Math.sin(camPhi), CENTER.z+camR*Math.cos(camTheta)*Math.cos(camPhi));
  camera.lookAt(CENTER);
}}
updateCam();

function setView(v){{
  document.querySelectorAll('.btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active'); rotating=false; document.getElementById('rot-btn').textContent='Auto Rotate';
  if(v==='front')  {{camR=26;camTheta=0;camPhi=0.18;}}
  else if(v==='side') {{camR=26;camTheta=Math.PI/2;camPhi=0.18;}}
  else if(v==='top')  {{camR=22;camTheta=0;camPhi=1.4;}}
  else              {{camR=22;camTheta=0.55;camPhi=0.32;}}
  updateCam();
}}
function toggleRotate(){{ rotating=!rotating; document.getElementById('rot-btn').textContent=rotating?'Stop':'Auto Rotate'; }}

canvas.addEventListener('mousedown',e=>{{isDrag=true;lastX=e.clientX;lastY=e.clientY;}});
canvas.addEventListener('mouseup',()=>{{isDrag=false;}}); canvas.addEventListener('mouseleave',()=>{{isDrag=false;}});
canvas.addEventListener('mousemove',e=>{{
  if(isDrag){{ camTheta-=(e.clientX-lastX)*0.007; camPhi=Math.max(0.05,Math.min(1.4,camPhi+(e.clientY-lastY)*0.007)); lastX=e.clientX; lastY=e.clientY; updateCam(); }}
  const rect=canvas.getBoundingClientRect();
  const mouse=new THREE.Vector2(((e.clientX-rect.left)/rect.width)*2-1, -((e.clientY-rect.top)/rect.height)*2+1);
  const ray=new THREE.Raycaster(); ray.setFromCamera(mouse,camera);
  const hits=ray.intersectObjects(meshObjects);
  if(hits.length){{
    const obj=hits[0].object;
    if(hoveredMesh!==obj){{
      if(hoveredMesh&&origColor!==null){{hoveredMesh.material.color.setHex(origColor);hoveredMesh.material.emissive&&hoveredMesh.material.emissive.set(0x000000);}}
      hoveredMesh=obj; origColor=obj.material.color.getHex();
      obj.material.color.setHex(0xffffff); obj.material.emissive=new THREE.Color(0x222222);
    }}
    tt.innerHTML=`<strong>${{obj.userData.product.name}}</strong><br>Category: ${{obj.userData.product.cat}}<br>Mode: ${{obj.userData.product.mode}}<br>Facings: ${{obj.userData.product.facings}}`;
    tt.style.display='block'; canvas.style.cursor='pointer';
  }} else {{
    if(hoveredMesh&&origColor!==null){{ hoveredMesh.material.color.setHex(origColor); if(hoveredMesh.material.emissive)hoveredMesh.material.emissive.set(0x000000); hoveredMesh=null;origColor=null; }}
    tt.style.display='none'; canvas.style.cursor='default';
  }}
}});
canvas.addEventListener('wheel',e=>{{ e.preventDefault(); camR=Math.max(8,Math.min(50,camR+e.deltaY*0.04)); updateCam(); }},{{passive:false}});

function animate(){{ requestAnimationFrame(animate); if(rotating){{camTheta+=0.005;updateCam();}} renderer.render(scene,camera); }}
animate();
</script>
</body>
</html>
"""
    components.html(html, height=height)

with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.6rem 0">
      <div style="font-size:18px;font-weight:700;color:#F9FAFB">Shelf Allocator</div>
      <div style="font-size:12px;color:#9CA3AF;margin-top:2px">MILP Optimizer — CBC Solver</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div style="font-size:11px;font-weight:600;color:#9CA3AF;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px">Solver Settings</div>', unsafe_allow_html=True)
    time_limit = st.slider("Time limit (seconds)", 20, 120, 60, 10)

    st.divider()
    run_btn   = st.button("Run Optimizer", type="primary", use_container_width=True)
    clear_btn = st.button("Clear Results", use_container_width=True)

if clear_btn and "result" in st.session_state:
    del st.session_state["result"]
    del st.session_state["result_time_limit"]
    st.rerun()

if "result_time_limit" in st.session_state:
    if st.session_state["result_time_limit"] != time_limit:
        del st.session_state["result"]
        del st.session_state["result_time_limit"]

st.markdown(f"""
<div class="top-bar">
  <div class="top-bar-icon">{icon(ICONS['layers'],'#60A5FA',22)}</div>
  <div>
    <p class="top-bar-title">MILP Shelf Display Allocator</p>
    <p class="top-bar-sub">Physical shelf allocation using Mixed-Integer Linear Programming
      &nbsp;·&nbsp; CBC Exact Solver &nbsp;·&nbsp;
      50 Products &nbsp;·&nbsp; 15 Shelf Locations</p>
  </div>
</div>""", unsafe_allow_html=True)
st.divider()

if run_btn:
    result_holder = [None]
    error_holder  = [None]

    def solver_thread():
        try:
            result_holder[0] = run_solver(data, time_limit=time_limit, verbose=False)
        except Exception as e:
            error_holder[0] = str(e)

    t = threading.Thread(target=solver_thread)
    t.start()

    st.markdown(f"""
    <div class="progress-wrap">
      <div class="progress-title">CBC Solver Running</div>
      <div class="progress-sub">
        Finding the optimal layout for 50 products across 15 shelves.
        Time limit: {time_limit} seconds.
      </div>
    </div>""", unsafe_allow_html=True)

    bar      = st.progress(0)
    status   = st.empty()
    start    = time.time()
    phases   = [(0.08, "Loading constraint matrix..."), (0.20, "Building LP relaxation..."), (0.40, "Running branch and bound..."), (0.65, "Pruning infeasible branches..."), (0.85, "Tightening bounds..."), (0.95, "Verifying solution...")]
    phase_idx = 0

    while t.is_alive():
        elapsed  = time.time() - start
        fraction = min(elapsed / time_limit, 0.97)
        bar.progress(fraction)

        if phase_idx < len(phases) and fraction >= phases[phase_idx][0]:
            status.markdown(f'<div style="font-size:12px;color:#6B7280;margin-top:-10px">{phases[phase_idx][1]} &nbsp; ({elapsed:.0f}s)</div>', unsafe_allow_html=True)
            phase_idx += 1
        time.sleep(0.4)

    t.join()
    bar.progress(1.0)
    status.markdown('<div style="font-size:12px;color:#059669;margin-top:-10px;font-weight:600">Solver complete</div>', unsafe_allow_html=True)
    time.sleep(0.4)
    bar.empty()
    status.empty()

    if error_holder[0]:
        st.error(f"Solver error: {error_holder[0]}")
    else:
        st.session_state["result"]            = result_holder[0]
        st.session_state["result_time_limit"] = time_limit

if "result" in st.session_state:
    result = st.session_state["result"]
    
    if result["solver_status"] in ("Optimal", "Integer Feasible") and not result["best_layout"].empty:
        best_layout   = result["best_layout"]
        improvement   = result["improvement"]
        cat_summary   = result["category_summary"]
        shelf_summary = result["shelf_summary"]
        rules_ok      = int(cat_summary["Within_Rules"].sum())
        rules_total   = len(cat_summary)
        gain          = improvement["utility_gain_pct"]

        sec_header("bar", "Key Results")
        gain_cls = "kpi-badge-pos" if gain >= 0 else "kpi-badge-neg"
        gain_str = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"

        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-icon-row">
              <div class="kpi-icon" style="background:#EFF6FF">{icon(ICONS['zap'],'#2563EB',16)}</div>
              <span class="kpi-tag">Utility Score</span>
            </div>
            <div class="kpi-val">{improvement['optimized_utility']:,.0f}</div>
            <div class="kpi-sub">Engineering weighted score</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-icon-row">
              <div class="kpi-icon" style="background:#F0FDF4">{icon(ICONS['trending'],'#16A34A',16)}</div>
              <span class="kpi-tag">Improvement</span>
            </div>
            <div class="kpi-val"><span class="{gain_cls}">{gain_str}</span></div>
            <div class="kpi-sub">vs current layout</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-icon-row">
              <div class="kpi-icon" style="background:#FFF7ED">{icon(ICONS['box'],'#EA580C',16)}</div>
              <span class="kpi-tag">Products Placed</span>
            </div>
            <div class="kpi-val">{improvement['products_placed']} / {improvement['total_products']}</div>
            <div class="kpi-sub">all products allocated</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-icon-row">
              <div class="kpi-icon" style="background:#F5F3FF">{icon(ICONS['check'],'#7C3AED',16)}</div>
              <span class="kpi-tag">Category Rules</span>
            </div>
            <div class="kpi-val">{rules_ok} / {rules_total}</div>
            <div class="kpi-sub">categories within range</div>
          </div>
        </div>""", unsafe_allow_html=True)

        sec_header("cube", "3D Shelf Viewer")
        st.caption("Drag to rotate  ·  Scroll to zoom  ·  Hover a product to see details")
        render_3d_viewer(best_layout, height=500)

        sec_header("grid", "2D Shelf Layout Map")
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        plot_shelf_layout(best_layout, ax=ax1)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        col_l, col_r = st.columns(2)
        with col_l:
            sec_header("bar", "Shelf Width Utilization")
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            plot_utilization(shelf_summary, ax=ax2)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        with col_r:
            sec_header("layers", "Category Balance")
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            plot_category_balance(cat_summary, ax=ax3)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        all_pass = rules_ok == rules_total
        sec_header("check", "Category Rule Compliance", badge=("All rules satisfied" if all_pass else f"{rules_ok}/{rules_total} satisfied", all_pass))
        display_cat = cat_summary.copy()
        display_cat["Status"] = display_cat["Within_Rules"].apply(lambda ok: "Within rules" if ok else "Violation")
        display_cat = display_cat.drop(columns=["Within_Rules"])
        st.dataframe(display_cat, use_container_width=True, hide_index=True)

        sec_header("table", "Full Optimized Layout")
        disp = best_layout[["Product_Name","Category","Display_Mode","Location_ID","Shelf_Level","Facings","Display_Units","Utility_Score"]].copy()
        disp.columns = ["Product","Category","Mode","Location","Level","Facings","Display Units","Utility Score"]
        disp["Utility Score"] = disp["Utility Score"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        sec_header("download", "Export")
        csv = best_layout.to_csv(index=False).encode("utf-8")
        st.download_button("Download Optimized Layout (CSV)", csv, "milp_optimized_layout.csv", "text/csv")
        
    else:
        st.error(f"❌ **Solver Status: {result['solver_status']}**\n\nThe mathematical model cannot find a solution. Even after relaxing limits, the combination of 50 products and strict Category Rules physically exceeds the remaining capacity of the 15 shelves.")
        
        # Display explicit errors instead of just disappearing!
        st.info("💡 **Why is this happening?**\n"
                "The solver is trapped between 'Category_Balance' minimums (which force it to add facings) and 'Shelf_Rack_Locations' weight/density limits (which forbid it from adding facings).")

else:
    sec_header("info", "Getting Started")
    st.info("Adjust the solver time limit in the sidebar and click Run Optimizer to begin.")

    sec_header("cube", "3D Shelf Viewer — Preview")
    st.caption("This will show your optimized layout after running the solver.")
    render_3d_viewer(None, height=420)