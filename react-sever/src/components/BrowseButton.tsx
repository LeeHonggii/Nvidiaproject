import React from 'react';

interface BrowseButtonProps {
  className: string;
  divClassName: string;
  onClick: () => void;
  onMouseEnter: () => void;
  overlapGroupClassName: string;
  property1: string;
  hasGroup?: boolean;
}

const BrowseButton: React.FC<BrowseButtonProps> = ({ className, divClassName, onClick, onMouseEnter, overlapGroupClassName, property1, hasGroup = true }) => {
  return (
    <button className={className} onClick={onClick} onMouseEnter={onMouseEnter}>
      <div className={divClassName}>Browse</div>
      {hasGroup && <div className={overlapGroupClassName}>Group</div>}
    </button>
  );
};

export default BrowseButton;
