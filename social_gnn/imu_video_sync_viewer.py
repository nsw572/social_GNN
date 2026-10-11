"""Create an offline HTML viewer synchronized to a video-time IMU clock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_signal(
    path: Path,
    *,
    axis: str,
    start_s: float,
    end_s: float,
    max_points: int,
) -> dict[str, object]:
    frame = pd.read_csv(path, usecols=["time", axis])
    time = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float)
    value = pd.to_numeric(frame[axis], errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(time)
        & np.isfinite(value)
        & (time >= start_s)
        & (time <= end_s)
    )
    time = time[keep]
    value = value[keep]
    if not len(time):
        raise ValueError(f"{path}: no finite {axis} samples inside [{start_s}, {end_s}]")
    if len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points).round().astype(int)
        time = time[indices]
        value = value[indices]
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    if np.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    margin = 0.05 * (maximum - minimum)
    return {
        "time": time.round(6).tolist(),
        "value": value.astype(float).round(6).tolist(),
        "yMin": minimum - margin,
        "yMax": maximum + margin,
        "axis": axis,
        "source": str(path),
    }


def build_viewer(
    *,
    video_path: Path,
    imu_a: Path,
    imu_b: Path,
    label_a: str,
    label_b: str,
    axis_a: str,
    axis_b: str,
    start_s: float,
    end_s: float,
    output_html: Path,
    max_points: int = 20000,
) -> Path:
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    for path in (video_path, imu_a, imu_b):
        if not path.is_file():
            raise FileNotFoundError(path)
    first = _load_signal(
        imu_a, axis=axis_a, start_s=start_s, end_s=end_s, max_points=max_points
    )
    second = _load_signal(
        imu_b, axis=axis_b, start_s=start_s, end_s=end_s, max_points=max_points
    )
    payload = {
        "start": start_s,
        "end": end_s,
        "signals": [
            {**first, "label": label_a, "color": "#1f77b4"},
            {**second, "label": label_b, "color": "#d62728"},
        ],
    }
    relative_video = Path(
        Path(video_path).resolve().relative_to(output_html.parent.resolve())
        if Path(video_path).resolve().is_relative_to(output_html.parent.resolve())
        else video_path.resolve()
    )
    video_uri = relative_video.as_posix()
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video + IMU synchronized viewer</title>
<style>
  :root {{ color-scheme: dark; font-family: Arial, sans-serif; }}
  body {{ margin: 0; background: #101216; color: #eef1f5; }}
  main {{ width: min(1500px, 96vw); margin: 16px auto 32px; }}
  header {{ display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }}
  h1 {{ font-size: 20px; margin: 0 0 10px; }}
  #clock {{ font: bold 18px Consolas, monospace; color: #ffd166; }}
  video {{ width: 100%; max-height: 62vh; background: black; display: block; }}
  .panel {{ margin-top: 12px; background: #181c22; border: 1px solid #343b46; }}
  .panel-title {{ padding: 7px 10px 0; font-weight: bold; }}
  canvas {{ width: 100%; height: 170px; display: block; cursor: crosshair; }}
  .hint {{ color: #9ca8b8; font-size: 13px; margin-top: 8px; }}
</style>
</head>
<body>
<main>
  <header><h1>A4_B4 video + two IMU acceleration axes</h1><span id="clock"></span></header>
  <video id="video" controls preload="metadata" src="{video_uri}"></video>
  <section class="panel"><div class="panel-title" id="title0"></div><canvas id="plot0"></canvas></section>
  <section class="panel"><div class="panel-title" id="title1"></div><canvas id="plot1"></canvas></section>
  <div class="hint">黄色竖线是当前视频时间；点击任一波形可跳转视频。CSV 时间戳直接按视频时间使用，没有额外偏移。</div>
</main>
<script>
const DATA={data_json};
const video=document.getElementById('video');
const clock=document.getElementById('clock');
const plots=DATA.signals.map((signal,index)=>({{signal,canvas:document.getElementById('plot'+index),base:null}}));
function nearest(signal,time){{
  let lo=0,hi=signal.time.length-1;
  while(lo<hi){{const mid=(lo+hi)>>1;if(signal.time[mid]<time)lo=mid+1;else hi=mid;}}
  if(lo>0 && Math.abs(signal.time[lo-1]-time)<Math.abs(signal.time[lo]-time))lo--;
  return signal.value[lo];
}}
function buildBase(plot){{
  const dpr=window.devicePixelRatio||1;
  const width=Math.max(400,Math.floor(plot.canvas.clientWidth*dpr));
  const height=Math.floor(170*dpr);
  plot.canvas.width=width; plot.canvas.height=height;
  const base=document.createElement('canvas'); base.width=width;base.height=height;
  const ctx=base.getContext('2d'); const s=plot.signal;
  ctx.fillStyle='#181c22';ctx.fillRect(0,0,width,height);
  ctx.strokeStyle='#303743';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){{const y=height*i/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(width,y);ctx.stroke();}}
  ctx.strokeStyle=s.color;ctx.lineWidth=Math.max(1,dpr);ctx.beginPath();
  for(let i=0;i<s.time.length;i++){{
    const x=(s.time[i]-DATA.start)/(DATA.end-DATA.start)*width;
    const y=height-(s.value[i]-s.yMin)/(s.yMax-s.yMin)*height;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }}
  ctx.stroke(); plot.base=base;
}}
function render(){{
  const absolute=DATA.start+video.currentTime;
  clock.textContent=`video ${{video.currentTime.toFixed(3)}} s | absolute ${{absolute.toFixed(3)}} s`;
  plots.forEach((plot,index)=>{{
    if(!plot.base || plot.base.width!==plot.canvas.width)buildBase(plot);
    const ctx=plot.canvas.getContext('2d');ctx.drawImage(plot.base,0,0);
    const x=(absolute-DATA.start)/(DATA.end-DATA.start)*plot.canvas.width;
    ctx.strokeStyle='#ffd166';ctx.lineWidth=2*(window.devicePixelRatio||1);ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,plot.canvas.height);ctx.stroke();
    document.getElementById('title'+index).textContent=`${{plot.signal.label}} · ${{plot.signal.axis}} = ${{nearest(plot.signal,absolute).toFixed(4)}}`;
  }});
  if(!video.paused && !video.ended)requestAnimationFrame(render);
}}
plots.forEach(plot=>plot.canvas.addEventListener('click',event=>{{
  const rect=plot.canvas.getBoundingClientRect();
  const fraction=Math.min(1,Math.max(0,(event.clientX-rect.left)/rect.width));
  video.currentTime=fraction*(DATA.end-DATA.start);render();
}}));
video.addEventListener('play',()=>requestAnimationFrame(render));
video.addEventListener('pause',render);video.addEventListener('seeked',render);video.addEventListener('loadedmetadata',render);
window.addEventListener('resize',()=>{{plots.forEach(p=>p.base=null);render();}});
plots.forEach(buildBase);render();
</script>
</body>
</html>"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--imu-a", type=Path, required=True)
    parser.add_argument("--imu-b", type=Path, required=True)
    parser.add_argument("--label-a", default="mouse A")
    parser.add_argument("--label-b", default="mouse B")
    parser.add_argument("--axis-a", choices=("ax", "ay", "az"), default="az")
    parser.add_argument("--axis-b", choices=("ax", "ay", "az"), default="az")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=20000)
    args = parser.parse_args()
    output = build_viewer(
        video_path=args.video,
        imu_a=args.imu_a,
        imu_b=args.imu_b,
        label_a=args.label_a,
        label_b=args.label_b,
        axis_a=args.axis_a,
        axis_b=args.axis_b,
        start_s=args.start_s,
        end_s=args.end_s,
        output_html=args.output_html,
        max_points=args.max_points,
    )
    print(output)


if __name__ == "__main__":
    main()
