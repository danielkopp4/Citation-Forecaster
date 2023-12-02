// src/textBox.js
import React from 'react';

function textBox() {
    return (
        <div>
        <input
            type="text"
            value={this.state.value}
            onChange={() => console.log(this.state.value)}
         />
         </div>
     );

}

export default textBox