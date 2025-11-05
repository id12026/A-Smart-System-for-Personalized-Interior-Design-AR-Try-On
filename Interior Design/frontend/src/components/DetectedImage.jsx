import { useRef, useState, useLayoutEffect } from "react";

const DetectedImage = ({ src, detections = [], isDarkMode = false, maxWidth = 680 }) => {
  const imgRef = useRef(null);
  const [sz, setSz] = useState({ nw: 0, nh: 0, w: 0, h: 0 });

  useLayoutEffect(() => {
    if (!imgRef.current) return;
    const el = imgRef.current;
    const update = () => {
      const rect = el.getBoundingClientRect();
      setSz((s) => ({ ...s, w: rect.width, h: rect.height }));
    };
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [src]);

  return (
    <div style={{ position: "relative", display: "inline-block", maxWidth }}>
      <img
        ref={imgRef}
        src={src}
        alt="detected"
        onLoad={(e) => {
          const el = e.currentTarget;
          requestAnimationFrame(() => {
            const rect = el.getBoundingClientRect();
            setSz({ nw: el.naturalWidth, nh: el.naturalHeight, w: rect.width, h: rect.height });
          });
        }}
        style={{ width: "100%", height: "auto", borderRadius: 16, boxShadow: "0 8px 30px rgba(0,0,0,0.2)" }}
      />
      {sz.nw > 0 && detections?.length > 0 && (
        <div style={{ position: "absolute", left: 0, top: 0, width: sz.w, height: sz.h, pointerEvents: "none" }}>
          {detections.map((d, i) => {
            const [x1,y1,x2,y2] = d.bbox || [];
            if ([x1,y1,x2,y2].some((v)=>typeof v!=="number")) return null;
            const sx = sz.w / sz.nw; const sy = sz.h / sz.nh;
            const left = x1 * sx, top = y1 * sy, width = (x2-x1) * sx, height = (y2-y1) * sy;
            return (
              <div key={i} style={{ position: "absolute", left, top, width, height, border: `2px solid ${isDarkMode?"#38bdf8":"#0ea5e9"}`, borderRadius: 6 }}>
                <div style={{ position: "absolute", left: 0, top: -20, background: isDarkMode?"#0f172a":"#0ea5e9", color: "#fff", fontSize: 12, padding: "2px 6px", borderRadius: 4 }}>
                  {d.label} {d.confidence? `${Math.round(d.confidence*100)}%`: ""}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default DetectedImage;


