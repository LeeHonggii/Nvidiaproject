import React from 'react';
import { useNavigate } from 'react-router-dom';
import './styles/style.css';
import backgroundVideo from './background.mp4';
import overlayImage from './overlay.png';

const Main: React.FC = () => {
  const navigate = useNavigate();

  const handleStartClick = () => {
    navigate('/upload');
  };

  return (
    <div className="main">
      <video className="background-video" autoPlay loop muted>
        <source src={backgroundVideo} type="video/mp4" />
      </video>
      <div className="overlay">
        <img src={overlayImage} alt="Overlay" />
      </div>
      <div className="content">
        <div className="div centered">나만의 영상을 생성해보세요</div>
        <p className="p centered">댄스 영상, 숏폼 컨텐츠, ...를 쉽고 빠르게 편집해보세요</p>
        <button className="start-button" onClick={handleStartClick}>
          시작하기
        </button>
      </div>
    </div>
  );
};

export default Main;
