import { CssBaseline, ThemeProvider } from "@mui/material";
import React from "react";

import { ChatPage } from "./pages/ChatPage";
import { workshopTheme } from "./theme";

export function App(): React.ReactElement {
  return (
    <ThemeProvider theme={workshopTheme}>
      <CssBaseline />
      <ChatPage />
    </ThemeProvider>
  );
}
