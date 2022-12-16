FROM rustlang/rust:nightly

ENV CARGO_UNSTABLE_SPARSE_REGISTRY=true

WORKDIR /server
COPY . ./
RUN cargo build --release
CMD cargo run --release