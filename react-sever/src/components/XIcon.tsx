import React from 'react';

interface XIconProps {
  className: string;
}

const XIcon: React.FC<XIconProps> = ({ className }) => {
  return (
    <div className={className}>
      X
    </div>
  );
};

export default XIcon;
