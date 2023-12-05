import React from 'react';
import { useNavigate } from "react-router-dom";

const NavBar = () => {
  // Initialize the navigate function from React Router DOM
  const navigate = useNavigate();

  // Function to navigate to the '/about' page
  const gotoAbout = () => {
    navigate('/about');
  };

  // Function to navigate to the home page ('/')
  const gotoHome = () => {
    navigate('/');
  };

  // Function to navigate to the '/timeline' page
  const goToTimeline = () => {
    navigate('/timeline');
  };

  return (
    <div>
      <div className='App-header'><h1>Citation Forecaster</h1></div>
      <div className="navbar">
        {/* Buttons for navigation */}
        <button onClick={gotoHome}>HomePage</button>
        <button onClick={gotoAbout}>About</button>
        <button onClick={goToTimeline}>Timeline</button>
      </div>
    </div>
  );
}

export default NavBar; // Exporting the NavBar component
