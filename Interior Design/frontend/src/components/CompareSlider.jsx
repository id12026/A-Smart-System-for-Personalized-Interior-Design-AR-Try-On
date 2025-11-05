import { useState } from "react";

const handleBg = {
  background: "#0ea5e9",
  borderRadius: 8,
  height: 6,
};

const thumbStyle = {
  width: 22,
  height: 22,
  background: "#0ea5e9",
  borderRadius: "50%",
  border: "2px solid white",
  boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
};

const CompareSlider = ({ beforeSrc, afterSrc }) => {
  const [pos, setPos] = useState(50);

  return (
    <div style={{ position: "relative", width: "100%", maxWidth: 680, margin: "0 auto" }}>
      <div style={{ position: "relative", width: "100%", paddingTop: "60%", borderRadius: 12, overflow: "hidden", boxShadow: "0 8px 30px rgba(0,0,0,0.2)" }}>
        <img src={beforeSrc} alt="before" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
        <div style={{ position: "absolute", inset: 0, width: `${pos}%`, overflow: "hidden" }}>
          <img src={afterSrc} alt="after" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <input
          type="range"
          min={0}
          max={100}
          value={pos}
          onChange={(e) => setPos(Number(e.target.value))}
          style={{ width: "100%", appearance: "none", background: "transparent" }}
        />
        <style>{`
          input[type=range]::-webkit-slider-runnable-track { height: 6px; ${Object.entries(handleBg).map(([k,v])=>`${k}:${v}`).join(';')} }
          input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; margin-top: -8px; ${Object.entries(thumbStyle).map(([k,v])=>`${k}:${v}`).join(';')} }
          input[type=range]::-moz-range-track { height: 6px; ${Object.entries(handleBg).map(([k,v])=>`${k}:${v}`).join(';')} }
          input[type=range]::-moz-range-thumb { ${Object.entries(thumbStyle).map(([k,v])=>`${k}:${v}`).join(';')} }
        `}</style>
      </div>
    </div>
  );
};

export default CompareSlider;


