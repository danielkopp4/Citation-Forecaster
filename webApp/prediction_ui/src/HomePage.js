// src/HomePage.js
import React from 'react';

function HomePage() {
  return (
    <div>
      <div className="navbar">
        <button onClick={() => console.log("Button 1 clicked!")}>Button 1</button>
        <button onClick={() => console.log("Button 2 clicked!")}>Button 2</button>
        <button onClick={() => console.log("Button 3 clicked!")}>Button 3</button>
        <button onClick={() => console.log("Button 4 clicked!")}>Button 4</button>
      </div>
      <div className="content">
        Welcome to the homepage!
      </div>
    </div>
  );
}

export default HomePage;