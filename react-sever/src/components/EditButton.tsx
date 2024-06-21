import React from "react";

interface EditButtonProps {
  className: string;
}

const EditButton: React.FC<EditButtonProps> = ({ className }) => {
  return (
    <button className={`edit-button-instance ${className}`}>
      편집하기
    </button>
  );
};

export default EditButton;
