import { Box, Paper, Typography } from "@mui/material";
import React from "react";

import { ChatMessage } from "../types/chat";

const USER_LABELS: Record<ChatMessage["role"], string> = {
  user: "You",
  assistant: "Assistant"
};

const BUBBLE_COLORS: Record<ChatMessage["role"], string> = {
  user: "primary.main",
  assistant: "background.paper"
};

const TEXT_COLORS: Record<ChatMessage["role"], string> = {
  user: "primary.contrastText",
  assistant: "text.primary"
};

const BORDER_RADII: Record<ChatMessage["role"], string> = {
  user: "24px 24px 4px 24px",
  assistant: "24px 24px 24px 4px"
};

interface ChatBubbleProps {
  message: ChatMessage;
}

export function ChatBubble({ message }: ChatBubbleProps): React.ReactElement {
  const createdAt = new Date(message.timestamp);
  const formattedTime = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit"
  }).format(createdAt);

  const justifyContent = message.role === "user" ? "flex-end" : "flex-start";
  const alignment = message.role === "user" ? "flex-end" : "flex-start";

  return (
    <Box sx={{ display: "flex", justifyContent, mb: 1.5 }}>
      <Box sx={{ maxWidth: "80%", display: "flex", flexDirection: "column" }}>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ alignSelf: alignment, mb: 0.5 }}
        >
          {USER_LABELS[message.role]} • {formattedTime}
        </Typography>
        <Paper
          elevation={0}
          sx={{
            bgcolor: BUBBLE_COLORS[message.role],
            color: TEXT_COLORS[message.role],
            borderRadius: BORDER_RADII[message.role],
            px: 2,
            py: 1.5
          }}
        >
          <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
            {message.content}
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
}
