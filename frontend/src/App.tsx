import { CssBaseline, ThemeProvider } from "@mui/material";
import React from "react";
import { WorkshopShell } from "./components/WorkshopShell";
import { workshopTheme } from "./theme";

export function App(): React.ReactElement {
  return (
    <ThemeProvider theme={workshopTheme}>
      <CssBaseline />
      <WorkshopShell />
    </ThemeProvider>
  );
}

