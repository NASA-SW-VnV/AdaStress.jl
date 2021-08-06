FROM julia
WORKDIR /app
COPY build-deps.jl .
RUN apt-get update && \
	apt-get install -y build-essential lcov && \
	julia build-deps.jl

