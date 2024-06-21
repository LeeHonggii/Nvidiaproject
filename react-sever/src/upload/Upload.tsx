import React from "react";
import Component from "../components/Component";
import "../styles/2_style.css";
import "../styles/2_component9_style.css";
import "../styles/2_loader1_style.css";
import { useNavigate } from "react-router-dom";

const Upload: React.FC = () => {
  const navigate = useNavigate();

  const handleAnalyze = async () => {
    try {
      navigate('/loading');  // Loading 페이지로 이동
    } catch (error) {
      console.error('Error during analysis:', error);
    }
  };

  return (
    <div className="page">
      <div className="div-2">
        <div className="text-wrapper-2">이용 가이드</div>
        <p className="p">
          교차편집을 원하는 영상을 첨부하고 편집하기 버튼을 눌러주세요. 잠시후 교차편집 영상이 자동으로 생성됩니다.
        </p>
        <div className="text-wrapper-2">영상 첨부</div>
        <p className="p">영상 파일을 드래그하거나 찾아보세요</p>
        <div className="components-container">
          <div className="row">
            <Component property1="default" />
            <Component property1="default" />
            <Component property1="default" />
          </div>
          <div className="row">
            <Component property1="default" />
            <Component property1="default" />
            <Component property1="default" />
          </div>
        </div>
        <button className="edit-button-instance" onClick={handleAnalyze}>
          편집하기
        </button>
      </div>
    </div>
  );
};

export default Upload;
