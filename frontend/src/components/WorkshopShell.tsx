import { AppBar, Box, Container, Toolbar, Typography } from "@mui/material";
import React from "react";

export function WorkshopShell(): React.ReactElement {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <AppBar position="static" color="primary" enableColorOnDark>
        <Toolbar>
          <Typography variant="h6" component="div">
            RAG Architecture Workshop
          </Typography>
        </Toolbar>
      </AppBar>
      <Container sx={{ flexGrow: 1, py: 6 }}>
        <Typography variant="h4" gutterBottom>
          Welcome
        </Typography>
        <Typography color="text.secondary">
          Follow the workshop instructions to implement Advanced RAG, Graph RAG,
          and Reflective RAG architectures. This placeholder UI will evolve as
          new capabilities are added in each branch.
        </Typography>
      </Container>
      <Box component="footer" sx={{ py: 2, textAlign: "center" }}>
        <Typography variant="body2" color="text.secondary">
          © {new Date().getFullYear()} RAG Architecture Workshop
        </Typography>
      </Box>
    </Box>
  );
}

