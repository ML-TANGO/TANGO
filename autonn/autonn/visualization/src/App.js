import React from 'react';

import {BrowserRouter as Router, Routes, Route} from 'react-router-dom'
import Flow from './components/Fullpageflow';
import axios from 'axios';
import Info from './components/page/Info'
import Abstract from './components/page/Abstract'
import Code from './components/page/Code'

class App extends React.Component {
  render() {
    console.log("Class App is here!!")
    return (
      <div>
            <Router>
                <Routes>
                    <Route path="/" exact element={<Flow/>}/>
                    <Route path="/info" exact element={<Flow />}/>
                </Routes>
            </Router>
        </div>
    );
  }
}

export default App;



// replace /api/running/, get ID and thier status
// var userid = ''
// var project_id = ''
// var model = ''
// var status = ''
// var info_id = ''

// axios.get('/api/info')
//   .then(function (response) {
//     var k = Object.keys(response.data).length
//     console.log(k)
//     var info = response.data[k-1]
//     info_id = info.id
//     userid = info.userid
//     project_id = info.project_id
//     model = info.model_type + info.model_size
//     status = info.model_viz
//   })
//   .catch(function (error) {
//     console.error('GET /api/info : Error = ', error.message)
//     info_id = 'unknown'
//     userid = 'unknown'
//     project_id = 'not assigned'
//     model = 'not selected'
//     status = 'not ready'
//   })
//   .then(function () {
//     console.log('id=', info_id, 'userid=', userid, ' project_id=', project_id)
//     console.log('model=', model, 'ready to load model=', status)
//   });


// const jsonData= require('./VGG16.json');
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
//var e = 0;
//for (var j=1; e === 1; j++){
//
//axios.delete('/api/running/'.concat(j).concat('/'))
// .then(function (response) {
//   // handle success
////  continue;
//})
// .catch(function (error) {
//   // handle error
//   console.log("ERORRR@@@@@@@");
//   e = 1;
// })
// .then(function () {
//   // always executed
// });
// }

//}
//
////노드와 엣지 삭제하기
// tenace note: unneccessary action, instead we need to get userid & project_id if they have
// axios.get('/api/running/')
//  .then(function (response) {
//     var k = Object.keys(response.data).length
//     console.log('kkkkkkkkkkkkkk', k)
//    // handle success
//     if (k > 0) {
//       // for (var j=0;j<k+1;j++){
//       for (var j=0;j<k;j++){
//         axios.delete('/api/running/'.concat(j).concat('/'))
//       }
//     }
//  })
//  .catch(function (error) {
//    // handle error
//   console.log('Error', error.message)
//  })
//  .then(function () {
//    // always executed
//  });


//for (var j=0; j<200; j++){
//
//axios.delete('/api/running/'.concat(j).concat('/'))
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
