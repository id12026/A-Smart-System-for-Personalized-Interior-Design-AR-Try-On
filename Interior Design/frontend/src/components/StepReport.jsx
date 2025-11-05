import { Steps, Typography } from "antd";

const { Title, Paragraph, Text } = Typography;

function extractSection(text, label) {
  try {
    if (typeof text !== "string") return "";
    const re = new RegExp(`${label}[:\n\r\-]*([\s\S]*?)(\n\n|$)`, "i");
    const m = text.match(re);
    if (!m || typeof m[1] !== "string") return "";
    return m[1].trim();
  } catch (_) {
    return "";
  }
}

const StepReport = ({ markdownText = "", isDarkMode = false, roomType = "" }) => {
  const styles = extractSection(markdownText, "styles|style recommendations|design style|recommendations");
  const furniture = extractSection(markdownText, "furniture|furnishings|items");
  const layout = extractSection(markdownText, "layout|placement|circulation");
  const colors = extractSection(markdownText, "colors|palette|hex");
  const budget = extractSection(markdownText, "budget|cost|time");

  const roomHints = {
    living: {
      layout: "Anchor seating around a focal point; keep 90cm paths clear.",
      furniture: "Sofa + accent chairs sized to room; low TV console.",
    },
    bedroom: {
      layout: "Bed headboard against main wall; 60cm clearance each side.",
      furniture: "Bed + 2 nightstands; compact wardrobe or dresser.",
    },
    kitchen: {
      layout: "Prefer work triangle; keep prep zones clutter‑free.",
      furniture: "Compact dining set or island stools as space allows.",
    },
    bathroom: {
      layout: "Dry/wet separation; optimize storage with niches.",
      furniture: "Vanity with drawers; mirrored cabinet for vertical storage.",
    },
  };
  const hint = roomHints[roomType] || {};

  const steps = [
    { title: "Style Direction", description: styles || "Apply the selected interior style consistently across the space." },
    { title: "Furniture Suggestions", description: furniture || hint.furniture || "Choose essential pieces and right sizes; avoid clutter." },
    { title: "Optimized Layout", description: layout || hint.layout || "Ensure clear circulation and balanced focal points." },
    { title: "Color Palette (HEX)", description: colors || "Use 60/30/10 rule with complementary tones." },
    { title: "Budget & Timeline", description: budget || "Provide rough cost in INR/USD and time estimate." },
  ];

  return (
    <div style={{ maxWidth: 720, margin: "16px auto", textAlign: "left" }}>
      <Title level={4} style={{ marginBottom: 12, color: isDarkMode ? "#e5e7eb" : "#111827" }}>Step-by-Step Recommendations</Title>
      <Steps
        direction="vertical"
        current={steps.length}
        items={steps.map((s) => ({ title: s.title, description: <Paragraph style={{ marginBottom: 8 }}>{s.description}</Paragraph> }))}
      />
      <Text style={{ color: isDarkMode ? "#a1a1aa" : "#6b7280" }}>
        Tip: Use these steps sequentially to implement the makeover efficiently.
      </Text>
    </div>
  );
};

export default StepReport;


