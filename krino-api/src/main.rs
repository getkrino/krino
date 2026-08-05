use figment::providers::{Env, Format, Toml};
use krino_api::{config::ApiConfig, server::create_router, state::AppState};
use std::net::SocketAddr;

#[tokio::main(worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    // Disable the global Rayon pool so worker-pool threads are the only Rayon threads.
    // Each worker builds its own isolated ThreadPool via ThreadPoolBuilder::build().
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();
    // Load config from file + env vars
    let config: ApiConfig = figment::Figment::new()
        .merge(Toml::file("krino-api.toml"))
        .merge(Env::prefixed("KRINO_"))
        .extract()?;

    // Initialize tracing (JSON in production, pretty in dev)
    init_tracing(&config);

    tracing::info!("Starting Krino API v{}", env!("CARGO_PKG_VERSION"));

    // Load models and build app state
    let state = AppState::load(config.clone())?;

    // Build router
    let app = create_router(state);

    // Bind and serve
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Krino API listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Krino API shut down cleanly");
    Ok(())
}

fn init_tracing(config: &ApiConfig) {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!(
            "{},ort::logging={}",
            config.logging.level, config.logging.ort_level
        ))
    });

    match config.logging.format.as_str() {
        "json" => {
            fmt().with_env_filter(filter).json().init();
        }
        _ => {
            fmt().with_env_filter(filter).pretty().init();
        }
    }

    tracing::info!(
        format = config.logging.format,
        level = config.logging.level,
        "Logging initialized"
    );
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    tracing::info!("Shutdown signal received");
}
