import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './styles/result.css';

const Result: React.FC = () => {
  const [videoUrl, setVideoUrl] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchVideo = async () => {
      try {
        const response = await fetch('http://localhost:5000/final-video');
        if (response.ok) {
          const videoBlob = await response.blob();
          const videoObjectUrl = URL.createObjectURL(videoBlob);
          setVideoUrl(videoObjectUrl);
        } else {
          console.error('Error fetching video:', response.statusText);
        }
      } catch (error) {
        console.error('Error fetching video:', error);
      }
    };

    fetchVideo();
  }, []);

  const handleDownloadAndRedirect = async () => {
    // 비디오 다운로드
    const link = document.createElement('a');
    link.href = videoUrl;
    link.download = 'final_video.mp4';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // 비디오 파일 삭제 요청
    try {
      const response = await fetch('http://localhost:5000/delete-videos', { method: 'DELETE' });
      if (!response.ok) {
        console.error('Error deleting videos:', response.statusText);
      }
    } catch (error) {
      console.error('Error deleting videos:', error);
    }

    // 메인 페이지로 이동
    navigate('/');
  };

  return (
    <div className="result-page">
      {videoUrl ? (
        <div className="video-container">
          <video controls src={videoUrl} width="800" />
          <button className="download-button" onClick={handleDownloadAndRedirect}>
            저장하기
          </button>
        </div>
      ) : (
        <p>Loading video...</p>
      )}
    </div>
  );
};

export default Result;
