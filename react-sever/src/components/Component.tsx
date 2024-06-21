import React, { useRef, useState, useEffect } from "react";
import PropTypes from "prop-types";
import "../styles/2_component9_style.css";
import UploadButton from "./UploadButton";
import fileUploadIconDefault from "../assets/file_upload_icon.png";
import videoIcon from "../assets/video_icon.png";  // import 비디오 아이콘
import xIcon from "../assets/x_icon.png";  // import X 아이콘
import frame41 from "../assets/Property 1=Frame 41.png";
import frame42 from "../assets/Property 1=Frame 42.png";
import frame43 from "../assets/Property 1=Frame 43.png";
import frame44 from "../assets/Property 1=Frame 44.png";
import { useNavigate } from "react-router-dom";

interface Props {
  property1: "variant-4" | "variant-2" | "variant-3" | "default";
  fileUploadIcon?: string;
}

const Component: React.FC<Props> = ({ property1, fileUploadIcon }) => {
  const [state] = useState({
    property1: property1 || "default",
  });
  const [uploading, setUploading] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [dragOver, setDragOver] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleInputChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    if (file) {
      await handleFileUpload(file);
    }
  };

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0] || null;
    if (file) {
      await handleFileUpload(file);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log(data);
      setUploadedFileName(file.name);
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (uploading) {
      interval = setInterval(() => {
        setCurrentFrame((prevFrame) => (prevFrame + 1) % 4);
      }, 500); // 0.5초마다 프레임 변경
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [uploading]);

  const handleRemoveFile = async () => {
    try {
      const response = await fetch('http://localhost:5000/delete', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: uploadedFileName }),
      });
      const data = await response.json();
      console.log(data);
      setUploadedFileName(null);
    } catch (error) {
      console.error('Error deleting file:', error);
    }
  };

  const renderLoader = () => {
    const frames = [frame41, frame42, frame43, frame44];
    return (
      <div className="loader">
        {frames.map((frame, index) => (
          <img
            key={index}
            src={frame}
            alt={`loading frame ${index + 1}`}
            className={`loader-frame frame${index + 1} ${
              index === currentFrame ? "show" : ""
            }`}
          />
        ))}
      </div>
    );
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  return (
    <div
      className={`component ${state.property1} ${dragOver ? "drag-over" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        accept="video/*"
        onChange={handleInputChange}
      />
      {uploading ? (
        renderLoader()
      ) : uploadedFileName ? (
        <div className="uploaded">
          <img src={videoIcon} alt="Video icon" className="video-icon" />
          <div className="file-name">{uploadedFileName}</div>
          <img src={xIcon} alt="Remove file" className="remove-icon" onClick={handleRemoveFile} />
        </div>
      ) : (
        <>
          <img
            className="file-upload-icon"
            alt="File upload icon"
            src={fileUploadIcon || fileUploadIconDefault}
          />
          <UploadButton onClick={() => fileInputRef.current?.click()} />
        </>
      )}
    </div>
  );
};

Component.propTypes = {
  property1: PropTypes.oneOf(["variant-4", "variant-2", "variant-3", "default"]).isRequired as unknown as PropTypes.Validator<"variant-4" | "variant-2" | "variant-3" | "default">,
  fileUploadIcon: PropTypes.string,
};

export default Component;
