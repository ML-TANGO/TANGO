import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import LayerList from './components/page/Layer';

function App() {
  const [isYolo, setIsYolo] = useState(false);
  const [modelType, setModelType] = useState("");

  // API를 통해 task 값에 따라 isYolo 값을 설정
  useEffect(() => {
    const fetchTaskType = async () => {
      try {
        // /api/info/에서 task 값을 가져옴
        const response = await axios.get('/api/info/');
        const task = response.data[response.data.length - 1].task;  // 첫 번째 데이터의 task 값 사용
        const modelType = response.data[response.data.length - 1].model_type;

        // task 값이 "detection"이면 isYolo를 true로 설정
        //setIsYolo(task === 'detection');
        setIsYolo(modelType === 'yolov9');
      } catch (error) {
        console.error('Failed to fetch task type:', error);
      }
    };

    fetchTaskType();
  }, []);

  return (
    <div>
      <Router>
        <Routes>
          <Route path="/" exact element={<LayerList isYolo={isYolo} />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;

