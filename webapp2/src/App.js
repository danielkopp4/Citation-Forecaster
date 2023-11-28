import React from "react";
import './App.css';

const getCitations = event => alert(event.target.id);
const button = [
  {
    name: 'Get Predicted Citations',
    action: "running algotithim, please wait"
  }
];

/*
future code to then show number of submissions used and also andles reset and deletion of submissions
*/
handleIncrement = (counter) => {
  const counters = [...this.state.counters];
  const index = counters.indexOf(counter);
  counters[index] = { ...counters[index] };
  counters[index].value++;
  this.setState({ counters });
};

handleDecrement = (counter) => {
  const counters = [...this.state.counters];
  const index = counters.indexOf(counter);
  counters[index] = { ...counters[index] };
  counters[index].value--;
  this.setState({ counters });
};

handleReset = () => {
  const counters = this.state.counters.map((c) => {
    c.value = 0;
    return c;
  });
  this.setState({ counters });
};

handleDelete = (counterId) => {
  const counters = this.state.counters.filter((c) => c.id !== counterId);
  this.setState({ counters });
};

handleRestart = () => {
  window.location.reload();
};
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