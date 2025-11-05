import { useMemo } from "react";

// Simple AR Try-On using <model-viewer> web component
// Requires model-viewer script in index.html

// Use a known-good public model with CORS + iOS usdz
const sampleModel = "https://modelviewer.dev/shared-assets/models/Astronaut.glb";
const iosSrc = "https://modelviewer.dev/shared-assets/models/Astronaut.usdz";

const ARTryOn = ({ isDarkMode }) => {
  const bg = isDarkMode ? "#0f0f0f" : "#ffffff";

  const style = useMemo(() => ({
    width: "100%",
    maxWidth: "680px",
    height: "460px",
    borderRadius: 12,
    overflow: "hidden",
    boxShadow: "0 8px 30px rgba(0,0,0,0.2)",
    background: bg,
    margin: "0 auto",
  }), [bg]);

  return (
    <div style={{ textAlign: "center" }}>
      {/* @ts-ignore - web component */}
      <model-viewer
        src={sampleModel}
        ios-src={iosSrc}
        ar
        ar-modes="webxr scene-viewer quick-look"
        camera-controls
        environment-image="neutral"
        exposure="1.0"
        shadow-intensity="1"
        auto-rotate
        style={style}
        poster="https://modelviewer.dev/shared-assets/models/placeholder.png"
        reveal="auto"
        interaction-prompt="auto"
      >
      </model-viewer>
      <div style={{ marginTop: 8, color: isDarkMode ? "#a1a1aa" : "#444" }}>
        Tip: On AR-capable devices, tap the AR icon to place the furniture.
      </div>
    </div>
  );
};

export default ARTryOn;


