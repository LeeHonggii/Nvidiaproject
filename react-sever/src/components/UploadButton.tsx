import React from "react";
import "../styles/upload_button_style.css"; // 스타일 파일 import

interface UploadButtonProps {
  onClick: () => void;
}

const UploadButton: React.FC<UploadButtonProps> = ({ onClick }) => {
  return (
    <button className="upload-button" onClick={onClick}>
      파일 선택하기
    </button>
  );
};

export default UploadButton;
