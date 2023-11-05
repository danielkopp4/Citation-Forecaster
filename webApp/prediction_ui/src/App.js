import logo from './logo.svg';
import './App.css';
import HomePage from './componenets/HomePage';
import textBox from './componenets/textBox'

function App() {
  return (
    <div className="App">
      <HomePage />
      {/* <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header> */
      }
      <textBox />{
        
      }
    </div>
  );
}

export default App;
