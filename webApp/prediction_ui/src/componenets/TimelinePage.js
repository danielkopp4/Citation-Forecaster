import React from 'react';
import './TimelinePage.css'; // Imports a CSS file for styling

function TimelinePage() {
  return (
    <div>
      <div className="timeline-container">
        <h1>Project Timeline</h1> {/* Renders the title of the timeline page */}
        <div className="timeline">
          {/* Timeline events */}
          <div className="timeline-event">
            <div className="event-content">
              <h2>Citation Dataset</h2> {/* Event title */}
              <p>Date: Jan 2023-present</p> {/* Event date */}
              {/*<p>Description of event 1...</p>*/}
            </div>
          </div>
          {/* Several more timeline events follow with similar structures */}
        </div>
      </div>
    </div>
  );
}

export default TimelinePage; // Exporting the TimelinePage component
