import { Box, Paper, Typography } from "@mui/material";
import React from "react";

import { ChatMessage } from "../types/chat";

interface ChatBubbleProps {
  message: ChatMessage;
}

const ROLE_LABELS: Record<ChatMessage["role"], string> = {
  user: "You",
  assistant: "Assistant"
};

const ROLE_COLORS: Record<ChatMessage["role"], string> = {
  user: "#DCF8C6",
  assistant: "#FFFFFF"
};

const ROLE_BORDER_RADIUS: Record<ChatMessage["role"], string> = {
  user: "24px 24px 4px 24px",
  assistant: "24px 24px 24px 4px"
};

export function ChatBubble({ message }: ChatBubbleProps): React.ReactElement {
  const createdAt = new Date(message.timestamp);
  const formattedTime = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit"
  }).format(createdAt);

  const align = message.role === "user" ? "flex-end" : "flex-start";

  return (
    <Box sx={{ display: "flex", justifyContent: align, mb: 1.5 }}>
      <Box sx={{ maxWidth: "80%", display: "flex", flexDirection: "column" }}>
        <Paper
          elevation={0}
          sx={{
            bgcolor: ROLE_COLORS[message.role],
            borderRadius: ROLE_BORDER_RADIUS[message.role],
            px: 2,
            py: 1.5,
            boxShadow: (theme) =>
              message.role === "assistant"
                ? theme.shadows[1]
                : "0px 1px 1px rgba(0,0,0,0.1)",
            display: "flex",
            flexDirection: "column",
            gap: 0.5,
            minWidth: "220px"
          }}
        >
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ fontWeight: 600 }}
          >
            {ROLE_LABELS[message.role]}
          </Typography>
          <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
            {message.content}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{
              alignSelf: "flex-end",
              fontSize: "0.7rem",
              opacity: 0.8
            }}
          >
            {formattedTime}
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
}
