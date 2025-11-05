import { useState, useRef, useLayoutEffect } from "react";
import { Upload, Typography, message } from "antd";
import { InboxOutlined, CloseCircleOutlined } from "@ant-design/icons";

const { Title } = Typography;
const { Dragger } = Upload;

const ImageUpload = ({ label, onImageChange, isDarkMode = false, onPreviewChange, detections = [], roomType }) => {
  const [preview, setPreview] = useState(null);
  const [imgSize, setImgSize] = useState({ naturalW: 0, naturalH: 0, w: 0, h: 0 });
  const imgRef = useRef(null);

  useLayoutEffect(() => {
    if (!imgRef.current) return;
    const el = imgRef.current;
    const update = () => {
      const rect = el.getBoundingClientRect();
      setImgSize((s) => ({ ...s, w: rect.width, h: rect.height }));
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [preview]);

  const uploadProps = {
    name: "file",
    multiple: false,
    maxCount: 1,
    accept: "image/*",
    showUploadList: false,
    beforeUpload: (file) => {
      const isImage = file.type.startsWith("image/");
      if (!isImage) {
        message.error("You can only upload image files!");
        return Upload.LIST_IGNORE;
      }

      const isLt10M = file.size / 1024 / 1024 < 10;
      if (!isLt10M) {
        message.error("Image must be smaller than 10MB!");
        return Upload.LIST_IGNORE;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        onImageChange(file);
        if (onPreviewChange) onPreviewChange(reader.result);
      };
      reader.readAsDataURL(file);
      return false;
    },
    onDrop: (e) => {
      console.log("Dropped files", e.dataTransfer.files);
    },
  };

  const handleRemove = () => {
    setPreview(null);
    onImageChange(null);
  };

  return (
    <div
      className="w-full transition-all duration-300 flex flex-col items-center"
      style={{ width: "100%", maxWidth: "500px", margin: "0 auto" }}
    >
      {label && (
        <Title level={5} style={{ marginBottom: "1rem", textAlign: "center" }}>
          {label}
        </Title>
      )}

      {preview ? (
        <div
          className="relative w-full flex justify-center items-center mt-4"
          style={{ maxWidth: "360px", margin: "0 auto" }}
        >
          <div className="relative" style={{ display: "inline-block" }}>
            <img
              src={preview}
              alt="Preview"
              ref={imgRef}
              onLoad={(e) => {
                const el = e.currentTarget;
                // wait a frame for layout to settle
                requestAnimationFrame(() => {
                  const rect = el.getBoundingClientRect();
                  setImgSize({ naturalW: el.naturalWidth, naturalH: el.naturalHeight, w: rect.width, h: rect.height });
                });
              }}
              style={{
                maxWidth: "100%",
                maxHeight: "480px",
                objectFit: "contain",
                borderRadius: 12,
                boxShadow: "0 4px 20px rgba(0,0,0,0.2)",
                display: "block",
                margin: "0 auto",
              }}
            />
            {/* Overlay detections */}
            {preview && detections && detections.length > 0 && imgSize.naturalW > 0 && (
              <div style={{ position: "absolute", left: 0, top: 0, width: imgSize.w, height: imgSize.h, pointerEvents: "none" }}>
                {detections.map((d, idx) => {
                  const [x1, y1, x2, y2] = d.bbox || [];
                  if ([x1,y1,x2,y2].some((v)=>typeof v !== 'number')) return null;
                  const sx = imgSize.w / imgSize.naturalW;
                  const sy = imgSize.h / imgSize.naturalH;
                  const left = x1 * sx;
                  const top = y1 * sy;
                  const width = (x2 - x1) * sx;
                  const height = (y2 - y1) * sy;
                  return (
                    <div key={idx} style={{ position: "absolute", left, top, width, height, border: `2px solid ${isDarkMode ? "#38bdf8" : "#0ea5e9"}`, borderRadius: 6 }}>
                      <div style={{
                        position: "absolute",
                        left: 0,
                        top: -20,
                        background: isDarkMode ? "#0f172a" : "#0ea5e9",
                        color: "#fff",
                        padding: "2px 6px",
                        borderRadius: 4,
                        fontSize: 11,
                      }}>
                        {d.label} {d.confidence ? `${Math.round(d.confidence*100)}%` : ""}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            <CloseCircleOutlined
              onClick={handleRemove}
              style={{
                position: "absolute",
                top: -10,
                right: -10,
                fontSize: 20,
                color: isDarkMode ? "#f87171" : "#ef4444",
                backgroundColor: isDarkMode ? "#1f1f1f" : "#ffffff",
                borderRadius: "50%",
                cursor: "pointer",
                boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
                zIndex: 10,
              }}
            />
            {roomType && (
              <div style={{ position: "absolute", left: 0, bottom: -28, fontSize: 12, color: isDarkMode ? "#a1a1aa" : "#374151" }}>
                Detected: <strong>{roomType}</strong>
              </div>
            )}
          </div>
        </div>
      ) : (
        <Dragger
          {...uploadProps}
          className="w-full max-w-xs p-4"
          style={{
            border: `1px dashed ${isDarkMode ? "#444" : "#d9d9d9"}`,
            borderRadius: 12,
            backgroundColor: isDarkMode ? "#1f1f1f" : "#fafafa",
          }}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined
              style={{ color: isDarkMode ? "#38bdf8" : "#1677ff" }}
            />
          </p>
          <p
            className="ant-upload-text"
            style={{ color: isDarkMode ? "#e5e5e5" : "#333" }}
          >
            Click or drag an image here to upload
          </p>
          <p
            className="ant-upload-hint"
            style={{ fontSize: 12, color: isDarkMode ? "#a1a1aa" : "#666" }}
          >
            Image only • Max size: 10MB
          </p>
        </Dragger>
      )}
    </div>
  );
};

export default ImageUpload;
