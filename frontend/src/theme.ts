import { createTheme } from "@mui/material/styles";

export const workshopTheme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#2A6AF7",
    },
    secondary: {
      main: "#FF6F3C",
    },
    background: {
      default: "#F5F7FB",
    },
  },
  typography: {
    fontFamily: "'Inter', 'Roboto', sans-serif",
    h4: {
      fontWeight: 600,
    },
  },
});

