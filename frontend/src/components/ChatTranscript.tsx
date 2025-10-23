import { Box } from "@mui/material";
import React, { useEffect, useRef } from "react";

import { ChatMessage } from "../types/chat";
import { ChatBubble } from "./ChatBubble";

interface ChatTranscriptProps {
  messages: ChatMessage[];
}

export function ChatTranscript({
  messages
}: ChatTranscriptProps): React.ReactElement {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [messages]);

  return (
    <Box
      ref={containerRef}
      sx={{
        flex: 1,
        overflowY: "auto",
        px: 2,
        py: 3,
        backgroundImage:
          "linear-gradient(180deg, rgba(42,106,247,0.05) 0%, rgba(255,111,60,0.05) 100%)"
      }}
    >
      {messages.map((message) => (
        <ChatBubble key={message.id} message={message} />
      ))}
    </Box>
  );
}
