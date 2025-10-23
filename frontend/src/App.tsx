import { CssBaseline, ThemeProvider } from "@mui/material";
import React from "react";
import { workshopTheme } from "./theme";
import { ChatPage } from "./pages/ChatPage";

export function App(): React.ReactElement {
  return (
    <ThemeProvider theme={workshopTheme}>
      <CssBaseline />
      <ChatPage />
    </ThemeProvider>
  );
}
