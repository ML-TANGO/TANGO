import React from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';
import App from './App';
import EditText from './EditText';
import EditTextarea from './EditTextarea';

axios.defaults.xsrfCookieName='csrftoken';
axios.defaults.xsrfHeaderName='X-CSRFTOKEN';

ReactDOM.render(<App />, document.getElementById('root'));
export { EditText, EditTextarea };
