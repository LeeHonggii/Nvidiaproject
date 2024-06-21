import React from 'react';

interface StartButtonProps {
  className: string;
  overlapGroupClassName: string;
  onClick: () => void;
}

const StartButton: React.FC<StartButtonProps> = ({ className, overlapGroupClassName, onClick }) => {
  return (
    <button className={className} onClick={onClick}>
      시작하기
    </button>
  );
};

export default StartButton;
