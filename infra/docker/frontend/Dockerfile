# Stage 1: Build the React application assets
FROM node:18-slim AS builder

WORKDIR /app

# Install dependencies using npm.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --silent

COPY frontend/ .
RUN npm run build

# Stage 2: Serve the compiled assets via a minimal web server
FROM nginx:1.27-alpine AS runner

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
