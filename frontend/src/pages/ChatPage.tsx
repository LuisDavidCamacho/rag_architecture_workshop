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
import React, { useCallback, useMemo, useState } from "react";

import { continueChat, embedDocuments, startChat } from "../api/client";
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

  const isEmbedding = embeddingState.status === "loading";

  const handleReset = useCallback(() => {
    setMessages([]);
    setChatId(null);
    setError(null);
    setEmbeddingState({ status: "idle" });
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

      setMessages((prev) => [
        ...prev,
        {
          id: createMessageId(),
          role: "user",
          content: text,
          timestamp
        }
      ]);
      setIsLoading(true);

      try {
        const payload = { query: text };
        const response = chatId
          ? await continueChat(chatId, payload)
          : await startChat(payload);

        setChatId(response.chat_id);

        setMessages((prev) => [
          ...prev,
          {
            id: createMessageId(),
            role: "assistant",
            content: response.response,
            timestamp: new Date().toISOString()
          }
        ]);
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Unexpected error communicating with the backend."
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
    return `Session • ${chatId}`;
  }, [chatId]);

  const handleEmbeddingComplete = useCallback(() => {
    setEmbeddingState({ status: "idle" });
  }, []);

  const handleEmbedDataset = useCallback(async () => {
    if (isEmbedding) {
      return;
    }

    const datasetName = "emails.csv"; // backend expects the bundled Enron corpus
    setEmbeddingState({ status: "loading", filename: datasetName });

    try {
      const response = await embedDocuments(datasetName);

      setEmbeddingState({
        status: "success",
        filename: datasetName,
        embeddedCount: response.embedded_documents
      });
    } catch (embeddingError) {
      const message =
        embeddingError instanceof Error
          ? embeddingError.message
          : "Embedding failed due to an unexpected error.";
      setEmbeddingState({
        status: "error",
        filename: datasetName,
        message
      });
    }
  }, [isEmbedding]);

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
            onClick={handleEmbedDataset}
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
            disabled={isLoading || isEmbedding}
          >
            Reset
          </Button>
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
            Successfully embedded <strong>{embeddingState.filename}</strong> — processed{" "}
            {embeddingState.embeddedCount} email chunks.
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
            Failed to embed{" "}
            {embeddingState.filename ? (
              <strong>{embeddingState.filename}</strong>
            ) : (
              "dataset"
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
            <CircularProgress size={22} />
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
