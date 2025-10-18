# Stage 1: Build the React application assets
FROM node:20-slim AS builder

WORKDIR /app

# Install dependencies based on the eventual package manifest (to be added later).
COPY frontend/package.json frontend/package-lock.json* frontend/pnpm-lock.yaml* frontend/yarn.lock* ./ 
RUN npm install --silent || pnpm install || yarn install

COPY frontend/ .
RUN npm run build || pnpm run build || yarn build

# Stage 2: Serve the compiled assets via a minimal web server
FROM nginx:1.27-alpine AS runner

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
