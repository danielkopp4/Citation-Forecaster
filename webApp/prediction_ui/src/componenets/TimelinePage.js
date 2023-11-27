import React from 'react';
import './TimelinePage.css';

function TimelinePage() {
  return (
    <div>
      <div className="navbar">
          <button onClick={() => console.log("Button 1 clicked!")}>HomePage</button>
          <button onClick={() => console.log("Button 2 clicked!")}>Timeline</button>
          <button onClick={() => console.log("Button 3 clicked!")}>About</button>
          <button onClick={() => console.log("Button 4 clicked!")}>Credits</button>
      </div>
      <div className="timeline-container">
        <h1>Project Timeline</h1>
        <div className="timeline">
          {/* Timeline events */}
          <div className="timeline-event">
            <div className="event-content">
              <h2>Citation Dataset</h2>
              <p>Date: Jan 2023-present</p>
              <p>Description of event 1...</p>
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>Initial Trained Model</h2>
              <p>Date: Jan 2023 - May 2023</p>
              <p>Description of event 2...</p>
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>Model Optimization</h2>
              <p>Date: Aug 2023 - present</p>
              <p>Description of event 2...</p>
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>Postgress DB</h2>
              <p>Date: Aug 2023 - present</p>
              <p>Description of event 2...</p>
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>React UI</h2>
              <p>Date: Aug 2023 - present</p>
              <p>Description of event 2...</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TimelinePage;
