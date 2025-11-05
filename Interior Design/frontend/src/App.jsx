import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  Select,
  Input,
  Button,
  Typography,
  Switch,
  Layout,
  ConfigProvider,
  theme,
  Row,
  Col,
  Divider,
} from "antd";
import { BulbFilled, BulbOutlined } from "@ant-design/icons";
import ImageUpload from "./components/ImageUpload";
import Footer from "./components/Footer";
import MarkdownCard from "./components/MarkdownCard";
import ReactMarkdown from "react-markdown";
import { Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip as ChartTooltip,
  Legend,
} from "chart.js";
import CompareSlider from "./components/CompareSlider";
import StepReport from "./components/StepReport";
import DetectedImage from "./components/DetectedImage";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import ReportPrint from "./components/ReportPrint";

ChartJS.register(ArcElement, ChartTooltip, Legend);

const { Header, Content } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;

function App() {
  const [homeImage, setHomeImage] = useState(null);
  const [backgroundColor, setBackgroundColor] = useState("#ffffff");
  const [foregroundColor, setForegroundColor] = useState("#000000");
  const [roomType, setRoomType] = useState("");
  const [style, setStyle] = useState("");
  const [history, setHistory] = useState([]);
  const [instructions, setInstructions] = useState("");
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem("darkMode");
    return saved ? JSON.parse(saved) : false;
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [successRate, setSuccessRate] = useState(null);
  const [beforeImage, setBeforeImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [genDetections, setGenDetections] = useState([]);
  const resultRef = useRef(null);
  const { defaultAlgorithm, darkAlgorithm } = theme;

  const extractPlantsInfo = (text) => {
    if (!text) return "";
    const lower = text.toLowerCase();
    const plantKeywords = ["plant", "potted", "greenery", "foliage", "herb", "flower"];
    if (!plantKeywords.some(k => lower.includes(k))) return "";
    const sentences = text.split(/[.!?]/);
    const plantSentences = sentences.filter(s => plantKeywords.some(k => s.toLowerCase().includes(k)));
    return plantSentences.join(". ").trim() || "";
  };

  const saveToDatabase = async (recommendationsText) => {
    try {
      const plantsInfo = extractPlantsInfo(recommendationsText);
      await axios.post(
        "http://127.0.0.1:8010/api/designs/save",
        {
          room_type: roomType,
          design_style: style,
          background_color: backgroundColor,
          foreground_color: foregroundColor,
          additional_instructions: instructions,
          interior_recommendations: recommendationsText,
          plants_info: plantsInfo,
        }
      );
      toast.success("Design saved successfully!");
    } catch (error) {
      console.error("Failed to save:", error);
    }
  };

  useEffect(() => {
    localStorage.setItem("darkMode", JSON.stringify(isDarkMode));
  }, [isDarkMode]);

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [result]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!homeImage) {
      toast.error("Please upload an image of your space");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("place_image", homeImage);
    formData.append("background_color", backgroundColor);
    formData.append("foreground_color", foregroundColor);
    formData.append("design_type", "interior");
    formData.append("room_type", roomType);
    formData.append("style", style);
    formData.append("instructions", instructions);

    try {
     const response = await axios.post("http://127.0.0.1:8010/api/try-on", formData, {
      headers: {
        "Content-Type": "multipart/form-data", 
      },
    });

     const newResult = {
        id: Date.now(),
        resultImage: response.data.image,
        text: response.data.text,
        success: response.data.success_rate,
        timestamp: new Date().toLocaleString(),
      };

      setResult(newResult);
      setSuccessRate(response.data.success_rate || null);
      setHistory((prev) => [newResult, ...prev]);
       if (response.data?.generated_analysis?.detections) {
         setGenDetections(response.data.generated_analysis.detections);
       } else {
         setGenDetections([]);
       }

      toast.success("Design generated successfully!");
      
      // Save to database (non-blocking)
      if (response.data.text) {
        saveToDatabase(response.data.text).catch(console.error);
      }
    } catch (error) {
      toast.error("Design generation failed");
    } finally {
      setLoading(false);
    }
  };

  // Auto analyze after image selected
  useEffect(() => {
    const run = async () => {
      if (!homeImage) return;
      try {
        const fd = new FormData();
        fd.append("place_image", homeImage);
        const res = await axios.post("http://127.0.0.1:8010/api/analyze", fd, { headers: { "Content-Type": "multipart/form-data" } });
        if (res.data?.room_type) setRoomType(res.data.room_type);
        if (Array.isArray(res.data?.detections)) setDetections(res.data.detections);
      } catch (err) {
        // silent fail
      }
    };
    run();
  }, [homeImage]);

  const textColor = "#1a1a1a";

  const pieData = successRate !== null ? {
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

  const downloadPdf = async () => {
    const target = document.getElementById("print-report");
    if (!target) return;
    // give Chart.js time to lay out when off-screen
    await new Promise((r) => setTimeout(r, 300));
    const canvas = await html2canvas(target, {
      scale: 2,
      useCORS: true,
      allowTaint: true,
      backgroundColor: "#ffffff",
      windowWidth: target.scrollWidth,
      windowHeight: target.scrollHeight,
    });

    const pdf = new jsPDF("p", "mm", "a4");
    const margin = 10; // mm
    const pageWidthMm = pdf.internal.pageSize.getWidth() - margin * 2;
    const pageHeightMm = pdf.internal.pageSize.getHeight() - margin * 2;

    // Convert mm page height to pixels using canvas aspect ratio
    const pxPerMm = canvas.width / pageWidthMm;
    const pageHeightPx = pageHeightMm * pxPerMm;

    let position = 0;
    let pageIndex = 0;

    while (position < canvas.height) {
      const sliceHeight = Math.min(pageHeightPx, canvas.height - position);

      // draw slice to a temp canvas
      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width = canvas.width;
      tmpCanvas.height = sliceHeight;
      const ctx = tmpCanvas.getContext("2d");
      ctx.drawImage(
        canvas,
        0,
        position,
        canvas.width,
        sliceHeight,
        0,
        0,
        canvas.width,
        sliceHeight
      );

      const imgData = tmpCanvas.toDataURL("image/png");
      const imgHeightMm = sliceHeight / pxPerMm;

      if (pageIndex > 0) pdf.addPage();
      pdf.addImage(imgData, "PNG", margin, margin, pageWidthMm, imgHeightMm);

      position += sliceHeight;
      pageIndex += 1;
    }

    pdf.save("interior-design-report.pdf");
  };

return (
  <ConfigProvider
    theme={{
      algorithm: isDarkMode ? darkAlgorithm : defaultAlgorithm,
      token: {
        colorPrimary: "#0ea5e9",
        borderRadius: 10,
      },
    }}
  >
    <Layout style={{ minHeight: "100vh", width: "100%", background: "rgba(255, 255, 255, 0.3)", backdropFilter: "blur(20px)" }}>
      <Header
        style={{
          background: "rgba(255, 255, 255, 0.5)",
          backdropFilter: "blur(15px)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "1.5rem 2rem",
          flexWrap: "wrap",
          borderBottom: "1px solid rgba(200, 150, 255, 0.4)",
          boxShadow: "0 4px 30px rgba(167, 139, 250, 0.15)",
        }}
      >
        <Title level={3} className="gradient-text" style={{ margin: 0, fontSize: "1.8rem", flex: 1, fontWeight: 800 }}>
          🏡 Virtual Home Designer
        </Title>
        <div style={{ marginLeft: "auto", display: "flex", gap: 12, alignItems: "center" }}>
          <Switch
            checked={isDarkMode}
            onChange={setIsDarkMode}
            checkedChildren={<BulbFilled />}
            unCheckedChildren={<BulbOutlined />}
          />
        </div>
      </Header>

      <Content style={{ padding: "2rem", width: "100%", minHeight: "calc(100vh - 80px)", background: "transparent" }}>
        <div style={{ width: "100%", maxWidth: "1400px", margin: "0 auto", padding: "0 2rem" }}>
          <Title level={2} className="anim-fade-in-up gradient-text" style={{ textAlign: "center", marginBottom: "3rem", fontSize: "2.5rem", fontWeight: 800 }}>
            Personalized Interior Design and AR Try‑On
          </Title>

          {/* FORM */}
          <form onSubmit={handleSubmit} className="dynamic-card depth-shadow" style={{ padding: "2rem", marginBottom: "3rem" }}>
            <Row gutter={[24, 24]} justify="center" style={{ flexWrap: "wrap" }}>
              <Col xs={24} sm={24} md={24} lg={12}>
                <ImageUpload
                  label="Upload Home Image"
                  onImageChange={setHomeImage}
                  onPreviewChange={setBeforeImage}
                  isDarkMode={isDarkMode}
                  detections={detections}
                  roomType={roomType}
                />
              </Col>

              <Col xs={24} sm={24} md={24} lg={12}>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  <div>
                    <Text style={{ color: textColor }}>Room Type</Text>
                    <Select
                      placeholder="Select room type"
                      style={{ width: "100%", marginTop: 4 }}
                      value={roomType}
                      onChange={setRoomType}
                    >
                      <Option value="living">Living Room</Option>
                      <Option value="bedroom">Bedroom</Option>
                      <Option value="kitchen">Kitchen</Option>
                      <Option value="bathroom">Bathroom</Option>
                    </Select>
                  </div>

                  <div>
                    <Text style={{ color: textColor }}>Design Style</Text>
                    <Select
                      placeholder="Select a style"
                      style={{ width: "100%", marginTop: 4 }}
                      value={style}
                      onChange={setStyle}
                    >
                      <Option value="modern">Modern</Option>
                      <Option value="minimalist">Minimalist</Option>
                      <Option value="rustic">Rustic</Option>
                      <Option value="bohemian">Bohemian</Option>
                      <Option value="classic">Classic</Option>
                    </Select>
                  </div>

                  <Row gutter={16}>
                    <Col xs={12}>
                      <Text style={{ color: textColor }}>Background Color</Text>
                      <Input
                        type="color"
                        value={backgroundColor}
                        onChange={(e) => setBackgroundColor(e.target.value)}
                        style={{
                          width: "100%",
                          height: "48px",
                          padding: "6px",
                          borderRadius: 8,
                          cursor: "pointer",
                        }}
                      />
                    </Col>
                    <Col xs={12}>
                      <Text style={{ color: textColor }}>Foreground Color</Text>
                      <Input
                        type="color"
                        value={foregroundColor}
                        onChange={(e) => setForegroundColor(e.target.value)}
                        style={{
                          width: "100%",
                          height: "48px",
                          padding: "6px",
                          borderRadius: 8,
                          cursor: "pointer",
                        }}
                      />
                    </Col>
                  </Row>
                </div>
              </Col>
            </Row>

            <div style={{ marginTop: 32 }}>
              <Text style={{ color: textColor }}>Additional Instructions</Text>
              <Input.TextArea
                rows={4}
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                style={{ width: "100%", marginTop: 8 }}
                placeholder="Example: Prefer warm lighting, eco-friendly materials, modern look, etc."
              />
            </div>

            <div style={{ textAlign: "center", marginTop: 40 }}>
              <Button
                type="primary"
                htmlType="submit"
                size="large"
                loading={loading}
                className="btn-enhanced"
                style={{ width: 220, height: 52, fontSize: 16 }}
              >
                {loading ? "Designing..." : "Generate Design"}
              </Button>
            </div>
          </form>

          {/* RESULT */}
          {result && (
            <div ref={resultRef} id="report-section" className="scale-in dynamic-card depth-shadow" style={{ marginTop: 64, textAlign: "center", padding: "2rem" }}>
              <Divider style={{ borderColor: "rgba(200, 150, 255, 0.3)" }} />
               <Title level={3} className="gradient-text" style={{ color: textColor, marginBottom: "2rem", fontSize: "2rem" }}>
                 Your Interior Designed Space
               </Title>
              {beforeImage ? (
                <>
                  <CompareSlider beforeSrc={beforeImage} afterSrc={result.resultImage} />
                  {/* Interior Design with detections */}
                  {genDetections?.length > 0 && (
                    <div style={{ marginTop: 32 }}>
                      <Title level={4} className="gradient-text" style={{ marginBottom: 16, fontSize: "1.5rem", fontWeight: 700 }}>Interior Design with Detections</Title>
                      <DetectedImage src={result.resultImage} detections={genDetections} isDarkMode={isDarkMode} />
                    </div>
                  )}
                </>
              ) : (
                genDetections?.length > 0 && (
                  <div style={{ marginTop: 32 }}>
                    <Title level={4} className="gradient-text" style={{ marginBottom: 16, fontSize: "1.5rem", fontWeight: 700 }}>Interior Design with Detections</Title>
                    <DetectedImage src={result.resultImage} detections={genDetections} isDarkMode={isDarkMode} />
                  </div>
                )
              )}
               {result.text && (
              <>
                 <StepReport markdownText={String(result.text || "")} isDarkMode={isDarkMode} />
                 <MarkdownCard text={String(result.text || "")} isDarkMode={isDarkMode} />
              </>
              )}

              {successRate !== null && (
                <div style={{ maxWidth: 360, margin: "24px auto" }}>
                  <Title level={5} style={{ color: textColor, marginBottom: 12 }}>Success Rate</Title>
                  <Pie data={pieData} />
                </div>
              )}

              <div style={{ marginTop: 16 }}>
                <Button onClick={downloadPdf} className="btn-enhanced glow-effect" size="large">
                  Download PDF Report
                </Button>
              </div>
              {/* Printable content for PDF generation */}
              <div style={{ position: "absolute", left: -99999, top: 0 }}>
                <ReportPrint
                  beforeImage={beforeImage}
                  resultImage={result?.resultImage}
                  markdownText={result?.text}
                  successRate={successRate}
                  roomType={roomType}
                  detections={detections}
                  genDetections={genDetections}
                  selections={{ roomType, style, backgroundColor, foregroundColor, instructions }}
                />
              </div>
            </div>
          )}

          {/* HISTORY */}
          {history.length > 0 && (
            <div style={{ marginTop: 80 }}>
              <Divider />
              <Title level={3} style={{ color: textColor, marginBottom: 32 }}>
                Previous Results
              </Title>
              <Row gutter={[24, 24]}>
                {history.map((item) => (
                  <Col xs={24} sm={12} md={8} key={item.id}>
                    <div
                      className="dynamic-card"
                      style={{
                        padding: 16,
                        height: "100%",
                      }}
                    >
                      <img
                        src={item.resultImage}
                        alt="Previous"
                        style={{
                          width: "100%",
                          borderRadius: 10,
                          marginBottom: 12,
                        }}
                      />
                      <ReactMarkdown
  children={item.text}
  style={{
    maxWidth: "680px",
    margin: "0 auto",
    padding: "1rem",
    background: "rgba(255, 255, 255, 0.5)",
    borderRadius: 12,
    overflowY: "auto",
    maxHeight: "400px",
    color: "#1a1a1a",
    lineHeight: 1.6,
  }}
/>
                      <Text
                        style={{
                          color: "#6b7280",
                          fontSize: 12,
                        }}
                      >
                        {item.timestamp}
                      </Text>
                    </div>
                  </Col>
                ))}
              </Row>
            </div>
          )}
        </div>
      </Content>

      <Footer isDarkMode={isDarkMode} />
      <ToastContainer theme={isDarkMode ? "dark" : "light"} />
    </Layout>
  </ConfigProvider>
);


}

export default App;