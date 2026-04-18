# Discterry — static `dist` only. No nginx: Traefik (Coolify) proxies to whatever port listens here.
# Build from repo root: `docker build -t discterry .`
#
# Coolify: context = repo root, Dockerfile = `./Dockerfile`, map Traefik → container port (default 3000).
# Optional env `PORT` overrides listen port (Coolify often sets this).

FROM node:22-alpine AS build
WORKDIR /app

COPY viz/discterry/package.json viz/discterry/package-lock.json ./
RUN npm ci

COPY viz/discterry/ ./
RUN npm run build

FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production

RUN npm install -g serve@14

COPY --from=build --chown=node:node /app/dist ./dist
USER node

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- "http://127.0.0.1:${PORT:-3000}/" >/dev/null || exit 1

CMD ["sh", "-c", "exec serve -s dist -p \"${PORT:-3000}\""]
