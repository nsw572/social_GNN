"""Component 1: interactive State Timeline Explorer HTML generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .state_atlas_viz_common import (
    AtlasViewTrial,
    iter_atlas_trials,
    mean_directions,
    raw_channel,
    transition_times,
)


DEFAULT_CHANNELS = (
    "hmm_state",
    "state_confidence",
    "centroid_distance",
    "pair_closing_speed",
    "movement_direction_alignment",
    "dynamic_quality",
    "pose_quality",
)


CHANNEL_LABELS = (
    ("hmm_state", "HMM state sequence"),
    ("state_confidence", "HMM posterior / confidence"),
    ("boundaries", "Bout / state boundaries"),
    ("centroid_distance", "centroid_distance"),
    ("nose_to_nose_distance", "nose_to_nose_distance"),
    ("source_nose_to_target_centroid_distance_0to1", "source_nose_to_target_centroid_distance 0->1"),
    ("source_nose_to_target_centroid_distance_1to0", "source_nose_to_target_centroid_distance 1->0"),
    ("source_faces_target_0to1", "source_faces_target 0->1"),
    ("source_faces_target_1to0", "source_faces_target 1->0"),
    ("head_direction_alignment", "head_direction_alignment"),
    ("source_radial_approach_speed_0to1", "source_radial_approach_speed 0->1"),
    ("source_radial_approach_speed_1to0", "source_radial_approach_speed 1->0"),
    ("pair_closing_speed", "pair_closing_speed"),
    ("movement_direction_alignment", "movement_direction_alignment"),
    ("dynamic_quality", "dynamic_quality (mean of directions)"),
    ("pose_quality", "pose_quality (mean of directions)"),
    ("pose_validity", "Pose validity indicators"),
    ("dynamic_diagnostics", "Dynamic sign-flip / jump diagnostics"),
)


def _masked(values: np.ndarray, valid_time_mask: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    output[~valid_time_mask] = np.nan
    return output


def build_timeline_payload(trial: AtlasViewTrial) -> dict[str, Any]:
    """Build a missingness-preserving browser payload for one frozen trial."""

    numeric = {
        "state_confidence": _masked(trial.state_confidence, trial.valid_time_mask),
        "centroid_distance": raw_channel(trial, 0, 0, 1),
        "nose_to_nose_distance": raw_channel(trial, 1, 0, 1),
        "source_nose_to_target_centroid_distance_0to1": raw_channel(trial, 2, 0, 1),
        "source_nose_to_target_centroid_distance_1to0": raw_channel(trial, 2, 1, 0),
        "source_faces_target_0to1": raw_channel(trial, 3, 0, 1),
        "source_faces_target_1to0": raw_channel(trial, 3, 1, 0),
        "head_direction_alignment": raw_channel(trial, 4, 0, 1),
        "source_radial_approach_speed_0to1": raw_channel(trial, 5, 0, 1),
        "source_radial_approach_speed_1to0": raw_channel(trial, 5, 1, 0),
        "pair_closing_speed": raw_channel(trial, 6, 0, 1),
        "movement_direction_alignment": raw_channel(trial, 7, 0, 1),
        "dynamic_quality": mean_directions(trial.dynamic_quality, trial.valid_time_mask),
        "pose_quality": mean_directions(trial.pose_quality, trial.valid_time_mask),
    }
    pose_validity_specs = (
        ("nose_to_nose_distance", 1, 0, 1),
        ("head_direction_alignment", 4, 0, 1),
        ("source_nose_to_target_centroid_distance 0->1", 2, 0, 1),
        ("source_nose_to_target_centroid_distance 1->0", 2, 1, 0),
        ("source_faces_target 0->1", 3, 0, 1),
        ("source_faces_target 1->0", 3, 1, 0),
    )
    pose_validity = {}
    for label, trait, source, target in pose_validity_specs:
        values = trial.edge_valid[:, source, target, trait].astype(np.float32)
        values[~trial.valid_time_mask] = np.nan
        pose_validity[label] = values

    diagnostics: dict[str, np.ndarray] = {}
    dynamic_specs = (
        ("source_radial_approach_speed 0->1", 0, 0, 1),
        ("source_radial_approach_speed 1->0", 0, 1, 0),
        ("pair_closing_speed", 1, 0, 1),
        ("movement_direction_alignment", 2, 0, 1),
    )
    for label, local_trait, source, target in dynamic_specs:
        diagnostics[f"{label} sign flip"] = _masked(
            trial.dynamic_sign_flip[:, source, target, local_trait],
            trial.valid_time_mask,
        )
        diagnostics[f"{label} absolute jump"] = _masked(
            trial.dynamic_abs_jump[:, source, target, local_trait],
            trial.valid_time_mask,
        )

    states = trial.predicted_state.astype(np.float64)
    states[~trial.valid_time_mask] = np.nan
    return {
        "trial_id": trial.trial_id,
        "time": trial.patch_center_s,
        "state": states,
        "boundaries": transition_times(trial),
        "numeric": numeric,
        "pose_validity": pose_validity,
        "dynamic_diagnostics": diagnostics,
        "time_min": float(trial.patch_start_s[0]),
        "time_max": float(trial.patch_end_s[-1]),
        "valid_points": int(trial.valid_time_mask.sum()),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            rounded = np.round(value.astype(np.float64), 6)
            return np.where(np.isfinite(rounded), rounded, None).tolist()
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _html_template() -> str:
    return r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>State Timeline Explorer — __TRIAL__</title>
  <script>__PLOTLY_JS__</script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; color: #202124; }
    header { padding: 12px 18px; background: #f5f7fa; border-bottom: 1px solid #d9dce1; }
    h1 { margin: 0 0 8px; font-size: 22px; }
    .controls { display: grid; grid-template-columns: repeat(6, minmax(120px, auto)); gap: 8px 14px; align-items: end; }
    .control label { display: block; font-size: 12px; font-weight: bold; margin-bottom: 3px; }
    input, select, button { font: inherit; padding: 5px; }
    button { cursor: pointer; }
    .channels { display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 3px 12px; margin-top: 10px; max-height: 185px; overflow-y: auto; }
    .channels label { font-size: 12px; white-space: nowrap; }
    #plot { width: 100%; min-height: 820px; }
    .note { font-size: 12px; color: #5f6368; margin-top: 7px; }
    @media (max-width: 1000px) { .controls { grid-template-columns: repeat(2, 1fr); } .channels { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <h1>State Timeline Explorer — <span id="trialTitle"></span></h1>
  <div class="controls">
    <div class="control"><label>Trial</label><select id="trialSelect">__TRIAL_OPTIONS__</select></div>
    <div class="control"><label>Start time (s)</label><input id="startTime" type="number" step="0.04"></div>
    <div class="control"><label>End time (s)</label><input id="endTime" type="number" step="0.04"></div>
    <div class="control"><label>Scale</label><select id="scaleMode"><option value="raw">Raw scale</option><option value="normalized">Per-channel normalized [0,1]</option><option value="robust">Robust z-score (median/IQR)</option></select></div>
    <div class="control"><label>Percentile clipping</label><span><input id="clipEnabled" type="checkbox"> <input id="clipLow" type="number" min="0" max="49" value="1" style="width:42px">–<input id="clipHigh" type="number" min="51" max="100" value="99" style="width:48px">%</span></div>
    <div class="control"><button id="applyButton">Apply</button> <button id="fullButton">Full trial</button></div>
  </div>
  <div class="channels" id="channelControls"></div>
  <div class="note">Missing values remain gaps. Scaling is computed separately for each displayed channel within the selected physical-time window. Plotly zoom/pan remains available.</div>
</header>
<div id="plot"></div>
<script>
const payload = __PAYLOAD__;
const channelLabels = __CHANNEL_LABELS__;
const defaultChannels = new Set(__DEFAULT_CHANNELS__);
const trialFiles = __TRIAL_FILES__;
const stateColors = ['#4E79A7','#F28E2B','#E15759','#76B7B2','#59A14F','#EDC948','#B07AA1','#FF9DA7','#9C755F','#BAB0AC','#6B8E23','#17BECF'];

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position), rest = position - base;
  return sorted[base + 1] === undefined ? sorted[base] : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}
function transformValues(values, mode, clip, low, high) {
  const finite = values.filter(v => v !== null && Number.isFinite(v)).slice().sort((a,b)=>a-b);
  if (!finite.length) return values.slice();
  let lo = -Infinity, hi = Infinity;
  if (clip) { lo = quantile(finite, low/100); hi = quantile(finite, high/100); }
  const clipped = values.map(v => v === null || !Number.isFinite(v) ? null : Math.max(lo, Math.min(hi, v)));
  const base = clipped.filter(v => v !== null).slice().sort((a,b)=>a-b);
  if (mode === 'raw') return clipped;
  if (mode === 'normalized') {
    const min = base[0], max = base[base.length-1], span = Math.max(max-min, 1e-12);
    return clipped.map(v => v === null ? null : (v-min)/span);
  }
  const median = quantile(base, .5), iqr = Math.max(quantile(base,.75)-quantile(base,.25), 1e-12);
  return clipped.map(v => v === null ? null : (v-median)/iqr);
}
function selectedKeys() { return [...document.querySelectorAll('.channelBox:checked')].map(x=>x.value); }
function rangeIndices(start, end) {
  let first = 0, last = payload.time.length;
  while (first < last && payload.time[first] < start) first++;
  while (last > first && payload.time[last-1] > end) last--;
  return [first,last];
}
function stateColorScale() {
  const result=[];
  stateColors.forEach((color,i)=>{ const a=i/12,b=(i+1)/12; result.push([a,color],[b,color]); });
  return result;
}
function render() {
  const start = Number(document.getElementById('startTime').value), end = Number(document.getElementById('endTime').value);
  if (!(end > start)) { alert('End time must be greater than start time.'); return; }
  const [first,last] = rangeIndices(start,end), time = payload.time.slice(first,last);
  const keys = selectedKeys(), mode=document.getElementById('scaleMode').value;
  const clip=document.getElementById('clipEnabled').checked, low=Number(document.getElementById('clipLow').value), high=Number(document.getElementById('clipHigh').value);
  const components=[];
  keys.forEach(key=>{
    if (key==='boundaries') return;
    if (key==='hmm_state') components.push({kind:'state',key,label:'HMM state'});
    else if (key==='pose_validity') components.push({kind:'validity',key,label:'Pose validity indicators'});
    else if (key==='dynamic_diagnostics') Object.keys(payload.dynamic_diagnostics).forEach(label=>components.push({kind:'numeric',key:'diag:'+label,label}));
    else components.push({kind:'numeric',key,label:channelLabels.find(x=>x[0]===key)[1]});
  });
  if (!components.length) components.push({kind:'state',key:'hmm_state',label:'HMM state'});
  const traces=[], layout={title:`${payload.trial_id}: ${start.toFixed(2)}–${end.toFixed(2)} s (${mode})`, margin:{l:210,r:30,t:55,b:55}, hovermode:'x unified', showlegend:false, xaxis:{range:[start,end],title:'Physical time (s)',rangeslider:{visible:false}}, height:Math.max(620,components.length*150), shapes:[]};
  const gap=Math.min(.008, .08/components.length);
  components.forEach((component,index)=>{
    const axisIndex=index+1, axisName=axisIndex===1?'y':'y'+axisIndex, layoutName=axisIndex===1?'yaxis':'yaxis'+axisIndex;
    const top=1-index/components.length-gap, bottom=1-(index+1)/components.length+gap;
    layout[layoutName]={domain:[Math.max(0,bottom),Math.min(1,top)],title:{text:component.label,font:{size:11}},automargin:true,zeroline:true};
    if (component.kind==='state') {
      traces.push({type:'heatmap',x:time,y:['state'],z:[payload.state.slice(first,last)],zmin:0,zmax:11,colorscale:stateColorScale(),showscale:false,hovertemplate:'time=%{x:.3f}s<br>state=%{z}<extra></extra>',yaxis:axisName,xaxis:'x',connectgaps:false});
      layout[layoutName].showticklabels=false;
    } else if (component.kind==='validity') {
      const labels=Object.keys(payload.pose_validity), z=labels.map(label=>payload.pose_validity[label].slice(first,last));
      traces.push({type:'heatmap',x:time,y:labels,z,zmin:0,zmax:1,colorscale:[[0,'#f2f2f2'],[.49,'#f2f2f2'],[.5,'#2ca02c'],[1,'#2ca02c']],showscale:false,hovertemplate:'%{y}<br>time=%{x:.3f}s<br>valid=%{z}<extra></extra>',yaxis:axisName,xaxis:'x',connectgaps:false});
      layout[layoutName].tickfont={size:8};
    } else {
      const raw=component.key.startsWith('diag:')?payload.dynamic_diagnostics[component.key.slice(5)].slice(first,last):payload.numeric[component.key].slice(first,last);
      const y=transformValues(raw,mode,clip,low,high);
      traces.push({type:'scattergl',mode:'lines',x:time,y,name:component.label,line:{width:1.2},connectgaps:false,hovertemplate:`${component.label}<br>time=%{x:.3f}s<br>value=%{y:.5g}<extra></extra>`,yaxis:axisName,xaxis:'x'});
    }
  });
  if (keys.includes('boundaries')) payload.boundaries.filter(x=>x>=start&&x<=end).forEach(x=>layout.shapes.push({type:'line',xref:'x',yref:'paper',x0:x,x1:x,y0:0,y1:1,line:{color:'rgba(60,60,60,.35)',width:1,dash:'dot'}}));
  Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false,scrollZoom:true});
}
function initialize() {
  document.getElementById('trialTitle').textContent=payload.trial_id;
  const start=payload.time_min, end=Math.min(payload.time_max,start+payload.default_window_s);
  document.getElementById('startTime').value=start.toFixed(2); document.getElementById('endTime').value=end.toFixed(2);
  const box=document.getElementById('channelControls');
  channelLabels.forEach(([key,label])=>{ const item=document.createElement('label'); const input=document.createElement('input'); input.type='checkbox'; input.className='channelBox'; input.value=key; input.checked=defaultChannels.has(key); item.appendChild(input); item.appendChild(document.createTextNode(' '+label)); box.appendChild(item); });
  const trialSelect=document.getElementById('trialSelect'); trialSelect.value=payload.trial_id; trialSelect.onchange=()=>{ window.location.href=trialFiles[trialSelect.value]; };
  document.getElementById('applyButton').onclick=render;
  document.getElementById('fullButton').onclick=()=>{ document.getElementById('startTime').value=payload.time_min.toFixed(2); document.getElementById('endTime').value=payload.time_max.toFixed(2); render(); };
  render();
}
initialize();
</script>
</body>
</html>'''


def write_timeline_html(
    trial: AtlasViewTrial,
    output_path: str | Path,
    *,
    trial_files: dict[str, str],
    default_window_s: float = 120.0,
    default_channels: Sequence[str] = DEFAULT_CHANNELS,
    include_plotlyjs: str = "inline",
) -> Path:
    if not np.isfinite(default_window_s) or default_window_s <= 0:
        raise ValueError("default_window_s must be finite and positive")
    known_channels = {key for key, _label in CHANNEL_LABELS}
    unknown_channels = set(default_channels) - known_channels
    if unknown_channels:
        raise ValueError(f"unknown default timeline channels: {sorted(unknown_channels)}")
    if include_plotlyjs == "inline":
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Plotly is required to generate the timeline explorer") from error
        plotly_js = get_plotlyjs()
    elif include_plotlyjs == "none":
        plotly_js = ""
    else:
        plotly_js = str(include_plotlyjs)
    payload = build_timeline_payload(trial)
    payload["default_window_s"] = float(default_window_s)
    options = "".join(
        f'<option value="{trial_id}">{trial_id}</option>' for trial_id in trial_files
    )
    html = (
        _html_template()
        .replace("__TRIAL__", trial.trial_id)
        .replace("__PLOTLY_JS__", plotly_js)
        .replace("__TRIAL_OPTIONS__", options)
        .replace("__PAYLOAD__", json.dumps(_json_ready(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        .replace("__CHANNEL_LABELS__", json.dumps(CHANNEL_LABELS, ensure_ascii=False))
        .replace("__DEFAULT_CHANNELS__", json.dumps(tuple(default_channels)))
        .replace("__TRIAL_FILES__", json.dumps(trial_files))
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def generate_timeline_explorer(
    atlas_root: str | Path,
    output_dir: str | Path,
    *,
    trial_ids: Sequence[str],
    default_window_s: float = 120.0,
    default_channels: Sequence[str] = DEFAULT_CHANNELS,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trial_files = {trial_id: f"{trial_id}_timeline.html" for trial_id in trial_ids}
    files = []
    for trial in iter_atlas_trials(atlas_root, tuple(trial_ids)):
        files.append(
            str(
                write_timeline_html(
                    trial,
                    output / trial_files[trial.trial_id],
                    trial_files=trial_files,
                    default_window_s=default_window_s,
                    default_channels=default_channels,
                ).resolve()
            )
        )
    index = f'''<!doctype html><html><head><meta charset="utf-8"><title>State Timeline Explorer</title><style>body{{font-family:Arial;margin:40px;max-width:800px}}select{{font-size:18px;padding:8px}}</style></head><body><h1>State Timeline Explorer</h1><p>Select one frozen trial. Each standalone page supports physical-time windows, Plotly zoom/pan, channel checkboxes, raw/min-max/robust scaling, and optional percentile clipping.</p><select id="trial"><option value="">Choose trial…</option>{''.join(f'<option value="{name}">{name}</option>' for name in trial_ids)}</select><script>const files={json.dumps(trial_files)};document.getElementById('trial').onchange=e=>{{if(e.target.value)location.href=files[e.target.value];}};</script></body></html>'''
    index_path = output / "index.html"
    index_path.write_text(index, encoding="utf-8")
    note = output / "README.md"
    note.write_text(
        "# State Timeline Explorer\n\nOpen `index.html`, choose a trial, then set a physical-time window and selected channels. Missing values are plotted as gaps. Raw, per-window min-max, and per-window median/IQR scales are available; percentile clipping is optional and display-only.\n",
        encoding="utf-8",
    )
    return {
        "index": str(index_path.resolve()),
        "trial_html": files,
        "usage_note": str(note.resolve()),
        "default_window_s": float(default_window_s),
        "default_channels": list(default_channels),
    }


__all__ = [
    "CHANNEL_LABELS",
    "DEFAULT_CHANNELS",
    "build_timeline_payload",
    "generate_timeline_explorer",
    "write_timeline_html",
]
