// src/HomePage.js
import React, {useState} from 'react';

function HomePage() {


  const [text, setText] = useState('');
  
  // Auto-resize the textarea based on content
  const handleTextChange = (event) => {
    const textarea = event.target;
    setText(textarea.value);
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
  };
  

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
          <div className="character-counter">{text.length} characters</div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;