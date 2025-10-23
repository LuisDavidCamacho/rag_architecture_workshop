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
  const [value, setValue] = useState("");

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const message = value.trim();
    if (!message) {
      return;
    }
    await onSend(message);
    setValue("");
  };

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{
        display: "flex",
        gap: 1,
        p: 1.5,
        borderTop: (theme) => `1px solid ${theme.palette.divider}`
      }}
    >
      <TextField
        fullWidth
        placeholder="Type your message"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        size="small"
        multiline
        maxRows={4}
        disabled={disabled}
      />
      <Tooltip title="Send message">
        <span>
          <IconButton
            color="primary"
            type="submit"
            disabled={disabled || value.trim().length === 0}
          >
            <SendIcon />
          </IconButton>
        </span>
      </Tooltip>
    </Box>
  );
}
