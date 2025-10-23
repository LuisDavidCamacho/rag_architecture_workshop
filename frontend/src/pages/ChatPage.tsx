import ChatIcon from "@mui/icons-material/Chat";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  Alert,
  AppBar,
  Backdrop,
  Box,
  Button,
  CircularProgress,
  Stack,
  Toolbar,
  Typography
} from "@mui/material";
import React, { useCallback, useMemo, useRef, useState } from "react";

import {
  continueChat,
  embedDocuments,
  startChat
} from "../api/ragClient";
import { ChatInput } from "../components/ChatInput";
import { ChatTranscript } from "../components/ChatTranscript";
import { ChatMessage } from "../types/chat";

type EmbeddingState =
  | { status: "idle" }
  | { status: "loading"; filename: string }
  | { status: "success"; filename: string; embeddedCount: number }
  | { status: "error"; filename?: string; message: string };

export function ChatPage(): React.ReactElement {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [embeddingState, setEmbeddingState] = useState<EmbeddingState>({
    status: "idle"
  });
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const isEmptyConversation = messages.length === 0;
  const isEmbedding = embeddingState.status === "loading";

  const handleReset = useCallback(() => {
    setMessages([]);
    setChatId(null);
    setError(null);
  }, []);

  const createMessageId = useCallback((): string => {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
    return Math.random().toString(36).slice(2);
  }, []);

  const handleSendMessage = useCallback(
    async (text: string) => {
      setError(null);
      const timestamp = new Date().toISOString();
      const userMessage: ChatMessage = {
        id: createMessageId(),
        role: "user",
        content: text,
        timestamp
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      try {
        const payload = { query: text };
        const response = chatId
          ? await continueChat(chatId, payload)
          : await startChat(payload);
        setChatId(response.chat_id);

        const assistantMessage: ChatMessage = {
          id: createMessageId(),
          role: "assistant",
          content: response.response,
          timestamp: new Date().toISOString()
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (thrown) {
        setError(
          thrown instanceof Error ? thrown.message : "Unexpected error occurred."
        );
      } finally {
        setIsLoading(false);
      }
    },
    [chatId, createMessageId]
  );

  const headerSubtitle = useMemo(() => {
    if (!chatId) {
      return "Start a conversation with the Advanced RAG backend";
    }

    return `Active session • ${chatId}`;
  }, [chatId]);

  const handleOpenEmbedDialog = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleEmbeddingComplete = useCallback(() => {
    setEmbeddingState({ status: "idle" });
  }, []);

  const handleFileSelection = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file) {
        return;
      }

      setEmbeddingState({ status: "loading", filename: file.name });

      try {
        const content = await file.text();
        const documents = parseDocuments(content);
        if (documents.length === 0) {
          throw new Error(
            "No textual content detected in the selected dataset. Please verify the file."
          );
        }

        const response = await embedDocuments({ documents });
        setEmbeddingState({
          status: "success",
          filename: file.name,
          embeddedCount: response.embedded_documents
        });
      } catch (embeddingError) {
        const message =
          embeddingError instanceof Error
            ? embeddingError.message
            : "Embedding failed due to an unexpected error.";
        setEmbeddingState({
          status: "error",
          filename: file.name,
          message
        });
      }
    },
    []
  );

  return (
    <Stack sx={{ height: "100vh", bgcolor: "grey.100" }}>
      <AppBar position="static" color="primary" enableColorOnDark>
        <Toolbar>
          <ChatIcon sx={{ mr: 1 }} />
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6">RAG Chat Playground</Typography>
            <Typography variant="body2" color="primary.contrastText">
              {headerSubtitle}
            </Typography>
          </Box>
          <Button
            variant="contained"
            color="secondary"
            startIcon={<CloudUploadIcon />}
            onClick={handleOpenEmbedDialog}
            disabled={isEmbedding}
            sx={{ mr: 1 }}
          >
            Embed Dataset
          </Button>
          <Button
            variant="outlined"
            color="inherit"
            startIcon={<RefreshIcon />}
            onClick={handleReset}
            disabled={(isLoading && !isEmptyConversation) || isEmbedding}
          >
            Reset
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.json,.csv"
            hidden
            onChange={handleFileSelection}
          />
        </Toolbar>
      </AppBar>

      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          maxWidth: 960,
          width: "100%",
          mx: "auto",
          my: 3,
          bgcolor: "common.white",
          boxShadow: 6,
          borderRadius: 3,
          overflow: "hidden"
        }}
      >
        {error ? (
          <Alert severity="error" sx={{ mx: 2, my: 2 }}>
            {error}
          </Alert>
        ) : null}

        {embeddingState.status === "success" ? (
          <Alert
            severity="success"
            sx={{ mx: 2, mt: 2 }}
            action={
              <Button color="inherit" size="small" onClick={handleEmbeddingComplete}>
                Back to chat
              </Button>
            }
          >
            Successfully embedded dataset{" "}
            <strong>{embeddingState.filename}</strong>. Documents processed:{" "}
            {embeddingState.embeddedCount}. You can now continue chatting.
          </Alert>
        ) : null}

        {embeddingState.status === "error" ? (
          <Alert
            severity="error"
            sx={{ mx: 2, mt: 2 }}
            action={
              <Button color="inherit" size="small" onClick={handleEmbeddingComplete}>
                Back to chat
              </Button>
            }
          >
            Failed to embed dataset{" "}
            {embeddingState.filename ? (
              <strong>{embeddingState.filename}</strong>
            ) : (
              "selected"
            )}
            . {embeddingState.message}
          </Alert>
        ) : null}

        <ChatTranscript messages={messages} />

        {isLoading ? (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              py: 1.5
            }}
          >
            <CircularProgress size={24} />
            <Typography variant="body2" sx={{ ml: 1.5 }} color="text.secondary">
              Waiting for response…
            </Typography>
          </Box>
        ) : null}

        <ChatInput onSend={handleSendMessage} disabled={isLoading || isEmbedding} />
      </Box>

      <Backdrop
        sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={isEmbedding}
      >
        <Stack spacing={2} alignItems="center">
          <CircularProgress color="inherit" />
          <Typography variant="body2">
            Generating embeddings for{" "}
            {embeddingState.status === "loading" ? embeddingState.filename : "dataset"}
            …
          </Typography>
        </Stack>
      </Backdrop>
    </Stack>
  );
}

function parseDocuments(content: string): string[] {
  try {
    const parsed = JSON.parse(content);
    if (Array.isArray(parsed)) {
      return parsed.map((item) =>
        typeof item === "string" ? item : JSON.stringify(item)
      );
    }
    if (typeof parsed === "object" && parsed !== null) {
      return Object.values(parsed).map((value) =>
        typeof value === "string" ? value : JSON.stringify(value)
      );
    }
  } catch {
    // ignore and fall back to text parsing
  }

  return content
    .split(/\n{2,}/)
    .map((segment) => segment.trim())
    .filter((segment) => segment.length > 0);
}
