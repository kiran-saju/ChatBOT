// script.js (updated)
document.addEventListener("DOMContentLoaded", () => {
  const chatIcon = document.getElementById("chat-icon");
  const chatContainer = document.getElementById("chat-container");
  const closeBtn = document.getElementById("close-btn");
  const overlay = document.getElementById("chat-overlay");
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("user-input");
  const chatBody = document.getElementById("chat-body");

  const API_BASE = "http://127.0.0.1:8000";

  let predefinedOptions = [];

  function openChat() {
    chatContainer.classList.add("visible");
    chatContainer.classList.remove("hidden");
    overlay.classList.add("visible");
    overlay.classList.remove("hidden");
    
    // Load predefined options when chat opens
    loadPredefinedOptions();
  }

  function closeChat() {
    chatContainer.classList.remove("visible");
    chatContainer.classList.add("hidden");
    overlay.classList.remove("visible");
    overlay.classList.add("hidden");
  }

  async function loadPredefinedOptions() {
    try {
      const response = await fetch(`${API_BASE}/options`);
      const data = await response.json();
      predefinedOptions = data.options;
      showPredefinedOptions();
    } catch (error) {
      console.error("Error loading options:", error);
      showDefaultOptions();
    }
  }

  function showPredefinedOptions() {
    if (chatBody.children.length === 0) {
      addMessage("Hello! I'm Ulearn Assistant. How can I help you today?", "bot");
      addMessage("Choose from common questions below or ask your own:", "bot");
      
      predefinedOptions.forEach(option => {
        addMessage(option.text, "bot", true, option.type);
      });
    }
  }

  function showDefaultOptions() {
    if (chatBody.children.length === 0) {
      addMessage("Hello! I'm Ulearn Assistant. How can I help you today?", "bot");
      addMessage("You can ask me about:", "bot");
      
      const defaultOptions = [
        "What courses do you offer?",
        "How much are the fees?",
        "What is the admission process?",
        "Do you offer scholarships?",
        "Free learning courses",
        "Connect to Counsellor"
      ];
      
      defaultOptions.forEach(option => {
        addMessage(option, "bot", true, "default");
      });
    }
  }

function addMessage(content, sender, isOption = false, optionType = "default") {
    const div = document.createElement("div");
    div.classList.add("message", sender === "bot" ? "bot-message" : "user-message");
    
    if (isOption) {
        div.classList.add("option-message");
        div.classList.add(optionType + "-option");
        div.style.cursor = "pointer";
        div.style.border = "1px solid #ddd";
        div.style.margin = "5px 0";
        div.style.padding = "8px 12px";
        div.style.borderRadius = "10px";
        div.style.transition = "all 0.2s ease";
        
        div.addEventListener("click", () => {
            userInput.value = content;
            sendMessage();
        });
        
        div.addEventListener("mouseenter", () => {
            div.style.background = "#f0f7ff";
            div.style.transform = "translateX(5px)";
        });
        
        div.addEventListener("mouseleave", () => {
            div.style.background = "";
            div.style.transform = "translateX(0)";
        });
        
        div.innerText = content; // Keep options as text only
    } else {
        // Check if content already contains HTML tags
        if (sender === "bot" && (content.includes('<a') || content.includes('<br') || content.includes('<p'))) {
            // Content already has HTML - use it as is
            div.innerHTML = content;
        }
        // Auto-detect and convert plain URLs to clickable links
        else if (sender === "bot" && content.includes('http')) {
            const urlRegex = /(https?:\/\/[^\s]+)/g;
            const formattedContent = content.replace(urlRegex, '<a href="$1" target="_blank" style="color: #ff4e9a; text-decoration: underline; word-break: break-all;">$1</a>');
            div.innerHTML = formattedContent;
        } else {
            div.innerText = content;
        }
    }
    
    chatBody.appendChild(div);
    chatBody.scrollTop = chatBody.scrollHeight;
}

  async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    
    addMessage(message, "user");
    userInput.value = "";

    // Show typing indicator
    const typingDiv = document.createElement("div");
    typingDiv.classList.add("message", "bot-message", "typing");
    typingDiv.id = "typing-indicator";
    typingDiv.innerHTML = "<div class='typing-dots'><span></span><span></span><span></span></div>";
    chatBody.appendChild(typingDiv);
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: message }),
      });
      
      const data = await res.json();
      
      // Remove typing indicator
      const typingIndicator = document.getElementById("typing-indicator");
      if (typingIndicator) {
        typingIndicator.remove();
      }
      
      // Show source indicator
      let responseText = data.response;
      if (data.source === "database_exact_match") {
        responseText += "\n\n✅ [From our FAQ database]";
      } else if (data.source === "ai_knowledge_base") {
        responseText += `\n\n🤖 [AI-powered response - Confidence: ${(data.confidence * 100).toFixed(0)}%]`;
      } else if (data.source === "fallback") {
        responseText += "\n\n💡 [General information - Contact support for details]";
      }
      
      addMessage(responseText, "bot");
      
    } catch (error) {
      // Remove typing indicator
      const typingIndicator = document.getElementById("typing-indicator");
      if (typingIndicator) {
        typingIndicator.remove();
      }
      
      addMessage("Sorry, I'm having trouble connecting right now. Please try again later.", "bot");
    }
  }

  // Event listeners
  chatIcon.addEventListener("click", openChat);
  closeBtn.addEventListener("click", closeChat);
  overlay.addEventListener("click", closeChat);
  sendBtn.onclick = sendMessage;
  
  userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });
});