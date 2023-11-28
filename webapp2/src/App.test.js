import React from "react";
import './App.css';

const getCitations = event => alert(event.target.id);
const button = [
  {
    name: 'Get Predicted Citations',
    action: "running algotithim, please wait"
  }
];

function App() {
  const greeting = "greeting";
  const displayAction = false;
  return(
    <div className="container">
      <h1 id={greeting}>Citation Predictor</h1>
      {displayAction && <p>I am writing JSX</p>}
      <ul>
        {
          button.map(name => (
            <li key={name.action}>
              <button
                onClick={getCitations}
              >
                <span role="img" aria-label={name.action} id={name.action}>{name.name}</span>
              </button>
            </li>
          ))
        }
      </ul>
    </div>
  )
}

export default App;
