import React, { useState } from 'react';
import LayerToggle from './LayerToggleNormal';
import LayerToggleYolo from './LayerToggleYolo';

const LayerToggleSwitcher = () => {
  const [isYolo, setIsYolo] = useState(false);

  const handleToggle = () => {
    setIsYolo(!isYolo);
  };

  return (
    <div>
      <button onClick={handleToggle}>
        {isYolo ? 'Switch to Normal' : 'Switch to Yolo'}
      </button>
      {isYolo ? <LayerToggleYolo /> : <LayerToggle />}
    </div>
  );
};

export default LayerToggleSwitcher;
