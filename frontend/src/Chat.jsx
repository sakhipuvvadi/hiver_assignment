import { useState } from "react";
import "./Chat.css";

function Chat() {

  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const sendQuery = async () => {

    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setResult(null);

    try {

      const response = await fetch(
        "http://127.0.0.1:8000/chat",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            query: query
          })
        }
      );

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();

      setResult(data);

    } catch (error) {

      console.error(error);

      setResult({
        response: "Unable to connect to the support server.",
        handling: "HUMAN"
      });

    } finally {

      setLoading(false);

    }
  };


  return (

    <div className="chat-container">

      <h1>Apple Support AI</h1>

      <p className="subtitle">
        Ask your Apple support question
      </p>


      <div className="input-section">

        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Example: My Apple Watch is not working..."
          rows="4"
        />

        <button
          onClick={sendQuery}
          disabled={loading}
        >

          {loading ? "Processing..." : "Send"}

        </button>

      </div>


      {result && (

        <div className="result-card">

          <div className="result-item">

            <strong>Cluster</strong>

            <span>
              {result.cluster}
            </span>

          </div>


          <div className="result-item">

            <strong>Intent</strong>

            <span>
              {result.intent}
            </span>

          </div>


          


          <div className="response-box">

            <strong>Response</strong>

            <p>
              {result.response}
            </p>

          </div>


          <div className="handling">

            <strong>Handling</strong>

            <span
              className={
                result.handling === "AUTOMATIC"
                  ? "automatic"
                  : "human"
              }
            >

              {result.handling}

            </span>

          </div>

        </div>

      )}

    </div>

  );
}

export default Chat;