import SendIcon from "@mui/icons-material/Send";
import { Box, IconButton, TextField, Tooltip } from "@mui/material";
import React, { useState } from "react";

interface ChatInputProps {
  onSend: (message: string) => Promise<void> | void;
  disabled?: boolean;
}

export function ChatInput({
  onSend,
  disabled = false
}: ChatInputProps): React.ReactElement {
  const [message, setMessage] = useState("");

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed) {
      return;
    }
    await onSend(trimmed);
    setMessage("");
  };

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{ display: "flex", gap: 1, p: 1, alignItems: "center" }}
    >
      <TextField
        fullWidth
        size="small"
        multiline
        maxRows={4}
        placeholder="Type a message"
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        disabled={disabled}
      />
      <Tooltip title="Send message">
        <span>
          <IconButton
            color="primary"
            type="submit"
            disabled={disabled || message.trim().length === 0}
          >
            <SendIcon />
          </IconButton>
        </span>
      </Tooltip>
    </Box>
  );
}
