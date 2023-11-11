// src/HomePage.js
import React, { useState } from 'react';

function HomePage() {
  const [text, setText] = useState('');
  const [notification, setNotification] = useState('');

  const handleTextChange = (event) => {
    const textarea = event.target;
    setText(textarea.value);
  };

  const handleSubmit = () => {
    const wordCount = text.trim().split(/\s+/).length;
    setNotification(`Word count: ${wordCount}. Servers were able to accept.`);
    // Here you could also send the text to a server if required
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
          {notification && <div className="notification">{notification}</div>}
        </div>
      </div>
    </div>
  );
}

export default HomePage;
