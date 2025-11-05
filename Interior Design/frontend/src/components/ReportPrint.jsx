import StepReport from "./StepReport";
import { Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip as ChartTooltip,
  Legend,
} from "chart.js";

ChartJS.register(ArcElement, ChartTooltip, Legend);

const sectionTitle = {
  fontSize: 16,
  fontWeight: 600,
  margin: "10px 0",
};

const caption = {
  fontSize: 12,
  color: "#64748b",
  textAlign: "center",
  marginTop: 6,
};

import { useState } from "react";

function DetectionFigure({ src, detections = [], roomType }) {
  const [sz, setSz] = useState({ nw: 0, nh: 0, w: 0, h: 0 });
  const maxW = 350;
  return (
    <div style={{ position: "relative", width: maxW }}>
      {src && (
        <img
          src={src}
          alt="detected"
          onLoad={(e) => {
            const el = e.currentTarget;
            // display width fits container; compute height proportionally
            const w = maxW;
            const h = (el.naturalHeight * w) / el.naturalWidth;
            setSz({ nw: el.naturalWidth, nh: el.naturalHeight, w, h });
          }}
          style={{ width: maxW, height: "auto", borderRadius: 8, border: "1px solid #e2e8f0" }}
        />
      )}
      {sz.nw > 0 && detections?.length > 0 && (
        <div style={{ position: "absolute", left: 0, top: 0, width: sz.w, height: sz.h, pointerEvents: "none" }}>
          {detections.map((d, i) => {
            const [x1, y1, x2, y2] = d.bbox || [];
            if ([x1,y1,x2,y2].some((v)=>typeof v!=="number")) return null;
            const sx = sz.w / sz.nw;
            const sy = sz.h / sz.nh;
            const left = x1 * sx, top = y1 * sy, width = (x2 - x1) * sx, height = (y2 - y1) * sy;
            return (
              <div key={i} style={{ position: "absolute", left, top, width, height, border: "2px solid #0ea5e9", borderRadius: 6 }}>
                <div style={{ position: "absolute", left: 0, top: -18, background: "#0ea5e9", color: "#fff", fontSize: 10, padding: "2px 4px", borderRadius: 4 }}>
                  {d.label} {d.confidence ? `${Math.round(d.confidence*100)}%` : ""}
                </div>
              </div>
            );
          })}
        </div>
      )}
      {roomType && (<div style={{ fontSize: 12, marginTop: 6, color: "#475569" }}>Detected room: <strong>{roomType}</strong></div>)}
    </div>
  );
}

const ReportPrint = ({ beforeImage, resultImage, markdownText, successRate, roomType, detections = [], genDetections = [], selections }) => {
  const pieData = typeof successRate === "number" ? {
    labels: ["Success", "Remaining"],
    datasets: [
      {
        data: [Math.round((successRate || 0) * 100), 100 - Math.round((successRate || 0) * 100)],
        backgroundColor: ["#22c55e", "#e5e7eb"],
        borderColor: ["#16a34a", "#cbd5e1"],
        borderWidth: 1,
      },
    ],
  } : null;
  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { position: "bottom" } },
  };

  return (
    <div id="print-report" style={{ width: 794, margin: "0 auto", padding: 20, background: "#ffffff", color: "#111827", boxShadow: "none" }}>
      <h2 style={{ textAlign: "center", marginBottom: 16 }}>Personalized Interior Design Report</h2>

      {/* Compare */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div>
          {beforeImage && (
            <img src={beforeImage} alt="Uploaded" style={{ width: "100%", borderRadius: 8, border: "1px solid #e2e8f0" }} />
          )}
          <div style={caption}>User uploaded image</div>
        </div>
        <div>
          {resultImage && (
            <img src={resultImage} alt="Interior Design" style={{ width: "100%", borderRadius: 8, border: "1px solid #e2e8f0" }} />
          )}
          <div style={caption}>Interior Design</div>
        </div>
      </div>

      {/* Detections for both */}
      <div style={{ height: 10 }} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div>
          <DetectionFigure src={beforeImage} detections={detections} roomType={roomType} />
          <div style={caption}>User uploaded image with detections</div>
        </div>
        <div>
          <DetectionFigure src={resultImage} detections={genDetections} roomType={null} />
          <div style={caption}>Interior Design with detections</div>
        </div>
      </div>

      {/* Selections */}
      <div style={{ height: 10 }} />
      {selections && (
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 8, padding: 12, marginTop: 6 }}>
          <div style={sectionTitle}>User Selections</div>
          <div style={{ fontSize: 13, lineHeight: 1.6 }}>
            <div><strong>Room Type:</strong> {selections.roomType || "-"}</div>
            <div><strong>Design Style:</strong> {selections.style || "-"}</div>
            <div><strong>Background Color:</strong> {selections.backgroundColor || "-"}</div>
            <div><strong>Foreground Color:</strong> {selections.foregroundColor || "-"}</div>
            {selections.instructions && (<div><strong>Additional Instructions:</strong> {selections.instructions}</div>)}
          </div>
        </div>
      )}

      {/* Compare (side-by-side already above) */}
      <div style={{ height: 8 }} />

      {/* Step by step */}
      <div style={sectionTitle}>Step-by-Step Recommendations</div>
      <StepReport markdownText={String(markdownText || "")} roomType={roomType} />

      {/* Interior recommendations raw markdown */}
      <div style={{ height: 8 }} />
      <div style={sectionTitle}>INTERIOR RECOMMENDATIONS</div>
      <div style={{
        border: "1px solid #e2e8f0",
        borderRadius: 8,
        padding: 12,
        fontSize: 13,
        lineHeight: 1.6,
        whiteSpace: "pre-wrap",
      }}>
        {String(markdownText || "")} 
      </div>

      {pieData && (
        <div style={{ marginTop: 12 }}>
          <div style={{ width: 220, height: 220 }}>
            <Pie data={pieData} options={pieOptions} redraw />
          </div>
        </div>
      )}
    </div>
  );
};

export default ReportPrint;


