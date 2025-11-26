import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import LayerList from './components/page/Layer';

function App() {
  const [hasActive, setHasActive] = useState(null); // null=로딩, true/false=판단됨
  const [isYolo, setIsYolo] = useState(false);
  const [modelType, setModelType] = useState("");

  // API를 통해 isYolo 값을 설정
  useEffect(() => {
    let timerId;

    const fetchActive = async () => {
      try {
        const { data } = await axios.get('/api/active-info/');
        if (data?.exists) {
          const decided = 
            !!data.model_type &&                // null/undefined/빈문자 방지
            data.model_type.trim() !== '' &&    // 공백만 있는 경우
            !/^not selected$/i.test(data.model_type); // 'not selected' 제외

          setHasActive(decided)

          if (decided) {
            setModelType(data.model_type ?? '');
            setIsYolo(Boolean(data.is_yolo));
          }
        } else {
          setHasActive(false);
        }
      } catch (e) {
        console.error('active-info fetch failed:', e);
        setHasActive(false); // 네트워크 오류 시 빈 상태 표시 (원하면 null로 유지해도 됨)
      }
    };

    fetchActive();                          // 즉시 1회
    timerId = setInterval(fetchActive, 2000); // 2초 폴링
    return () => clearInterval(timerId);
  }, []);

  useEffect(() => {
    console.log('state updated =>', modelType, isYolo)
  })

  // ---- 여기부터 조건부 렌더 ----
  if (hasActive === null) {
    return <div style={{ padding: 16 }}>불러오는 중…</div>;
  }

  if (hasActive !== true) {
    return <div style={{ padding: 16 }}>활성 작업이 없습니다. 작업이 시작되면 자동으로 갱신됩니다.</div>;
  }

  return (
    <div>
      <Router>
        <Routes>
          <Route 
            path="/"
            element={
              <LayerList
                key={`ll-${Number(isYolo)}`}
                isYolo={isYolo}
                onChangeIsYolo={setIsYolo} 
              />
            } 
          />
        </Routes>
      </Router>
    </div>
  );
}

export default App;

