import React from 'react';
import './TimelinePage.css';

function TimelinePage() {
  return (
    <div>
      
      <div className="timeline-container">
        <h1>Project Timeline</h1>
        <div className="timeline">
          {/* Timeline events */}
          <div className="timeline-event">
            <div className="event-content">
              <h2>Citation Dataset</h2>
              <p>Date: Jan 2023-present</p>
              {/*<p>Description of event 1...</p>*/}
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>Initial Trained Model</h2>
              <p>Date: Jan 2023 - May 2023</p>
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>Model Optimization</h2>
              <p>Date: Aug 2023 - present</p>
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>Postgress DB</h2>
              <p>Date: Aug 2023 - present</p>
            </div>
          </div>
          <div className="timeline-event">
            <div className="event-content">
              <h2>React UI</h2>
              <p>Date: Aug 2023 - present</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TimelinePage;
