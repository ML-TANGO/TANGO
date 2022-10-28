import React from 'react';

import {BrowserRouter as Router, Routes, Route} from 'react-router-dom'
import Flow from './components/Fullpageflow';
import axios from 'axios';

class App extends React.Component {
  render() {
    return (
      <div>
            <Router>
                <Routes>
                    <Route path="/" exact element={<Flow/>}/>
                </Routes>
            </Router>
        </div>
    );
  }
}

export default App;
const jsonData= require('./VGG16.json');
// console.log(jsonData.node[0].layer)


//for (var i=0; i<Object.keys(jsonData.node).length; i++){
//    axios.post("/api/node/",{
//        order: jsonData.node[i].order,
//        layer: jsonData.node[i].layer,
//        parameters: jsonData.node[i].parameters
//    }).then(function(response){
//        console.log(response)
//    }).catch(err=>console.log("error", jsonData.node[i].parameters, err));
//};
//
//for (var j=0; j<Object.keys(jsonData.edge).length; j++){
// axios.post("/api/edge/",{
//        id: jsonData.edge[j].id,
//        prior: jsonData.edge[j].prior,
//        next: jsonData.edge[j].next
//    }).then(function(response){
//        console.log(response)
//    }).catch(err=>console.log(err));
//};

//노드와 엣지 삭제하기
//for (var j=0; j<Object.keys(jsonData.node).length; j++){
//
//axios.delete('/api/node/'.concat(j))
//  .then(function (response) {
//    // handle success
//  })
//  .catch(function (error) {
//    // handle error
//  })
//  .then(function () {
//    // always executed
//  });
//}

// 노드와 엣지 삭제하기
//for (var j=0; j<60; j++){
//
//axios.delete('/api/node/'.concat(j).concat('/'))
// .then(function (response) {
//   // handle success
// })
// .catch(function (error) {
//   // handle error
// })
// .then(function () {
//   // always executed
// });
//}
//
////노드와 엣지 삭제하기
//for (var j=0; j<60; j++){
//
//axios.delete('/api/edge/'.concat(j).concat('/'))
// .then(function (response) {
//   // handle success
// })
// .catch(function (error) {
//   // handle error
// })
// .then(function () {
//   // always executed
// });
//}