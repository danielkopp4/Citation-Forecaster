import React from 'react';
import { useNavigate } from "react-router-dom";

const NavBar= () =>{
    const navigate = useNavigate();

    const gotoAbout = () => {
        navigate('/about');
      };

      const gotoHome = () => {
        navigate('/');
      };
      const goToTimeline = () => {
        navigate('/timeline');
      };
    return (
    <div>
        
        <div className='App-header'><h1>Citation Forcaster</h1></div>
        <div className="navbar">
            <button onClick={gotoHome}>HomePage</button>
            <button onClick={gotoAbout}>About</button>
            <button onClick={goToTimeline}>Timeline</button>
        </div>
    </div>
    );
}
export default NavBar;