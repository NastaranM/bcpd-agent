# Stage 2: package the agent so it runs the same on any machine or cloud service.
#   docker build -t bcpd-agent .
#   docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY bcpd-agent --yes "Did UNRATE change regime since 2000?"
FROM python:3.11-slim

# git is needed because bcpd is installed from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals
RUN pip install --no-cache-dir .

# Run as a non-root user: a small, standard security control.
RUN useradd --create-home agent && chown -R agent /app
USER agent

ENTRYPOINT ["bcpd-agent"]
