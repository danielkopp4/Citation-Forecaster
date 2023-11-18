// src/HomePage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';


function HomePage() {
  const [text, setText] = useState('');
  const [notification, setNotification] = useState('');
  const [message, setMessage] = useState('');

  const handleTextChange = (event) => {
    const textarea = event.target;
    setText(textarea.value);
  };
  const [responseMessage, setResponseMessage] = useState('');
  const handleSubmit = async () => {
    try {
      const response = await axios.post('/submit-text', { text });
      setResponseMessage(response.data.message);
    } catch (error) {
      console.error('Error submitting text:', error);
      setResponseMessage('Error submitting text');
    }
  };

  useEffect(() => {
    fetch("http://localhost:3001/api")
      .then((res) => res.json())
      .then((data) => setMessage(data.message));
  }, []);

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
        <div className="textbox-container">
          <textarea
            placeholder="Type your paragraph here..."
            value={text}
            onChange={handleTextChange}
            className="cool-textbox"
            rows={3}
          />
          <button onClick={handleSubmit} className="submit-button">Submit</button>
          {responseMessage && <div className="notification">Server: {responseMessage}</div>}
        </div>
      </div>
    </div>
  );
}

export default HomePage;