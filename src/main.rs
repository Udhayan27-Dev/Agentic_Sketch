mod processor;
mod samples;

use axum::{
    extract::{DefaultBodyLimit, Multipart},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use image::{DynamicImage, ImageFormat};
use processor::{BoundingBox, ImageProcessor, ProcessOptions};
use serde::{Deserialize, Serialize};
use std::{io::Cursor, net::SocketAddr};
use tower_http::cors::{Any, CorsLayer};

#[derive(Deserialize)]
struct ProcessJsonRequest {
    image_base64: String,
    options: ProcessOptions,
}

#[derive(Deserialize)]
struct IsolateJsonRequest {
    image_base64: String,
}

#[derive(Serialize)]
struct ProcessJsonResponse {
    image_base64: String,
    width: u32,
    height: u32,
    processing_time_ms: u128,
    bbox: Option<BoundingBox>,
}

#[tokio::main]
async fn main() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/process", post(process_handler))
        .route("/api/process_multipart", post(process_multipart_handler))
        .route("/api/isolate", post(isolate_handler))
        .route("/api/sample/:name", get(sample_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB limit
        .layer(cors);

    let port = 8080;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("--------------------------------------------------");
    println!("Engineering Drawing Background Remover (Rust)");
    println!("Server running on: http://127.0.0.1:{}", port);
    println!("--------------------------------------------------");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener");

    axum::serve(listener, app).await.expect("Server failed to run");
}

async fn index_handler() -> Html<String> {
    if let Ok(content) = std::fs::read_to_string("static/index.html") {
        Html(content)
    } else {
        Html(include_str!("../static/index.html").to_string())
    }
}

async fn process_handler(
    Json(payload): Json<ProcessJsonRequest>,
) -> Result<Json<ProcessJsonResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    let base64_data = if let Some(idx) = payload.image_base64.find(',') {
        &payload.image_base64[idx + 1..]
    } else {
        &payload.image_base64
    };

    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid base64: {}", e)))?;

    let dynamic_img = image::load_from_memory(&bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to decode image: {}", e)))?;
    let width = dynamic_img.width();
    let height = dynamic_img.height();

    // Process image with Rust background removal engine
    let output_img = ImageProcessor::process(&dynamic_img, &payload.options);

    // Compute bounding box of active drawing strokes
    let bbox = ImageProcessor::compute_bbox(&output_img);

    // Encode to PNG
    let mut png_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut png_bytes);
    DynamicImage::ImageRgba8(output_img)
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to encode PNG: {}", e)))?;

    let out_base64 = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png_bytes)
    );

    let duration = start.elapsed().as_millis();

    Ok(Json(ProcessJsonResponse {
        image_base64: out_base64,
        width,
        height,
        processing_time_ms: duration,
        bbox,
    }))
}

async fn process_multipart_handler(
    mut multipart: Multipart,
) -> Result<Response, (StatusCode, String)> {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut options = ProcessOptions::default();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "image" {
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Field read error: {}", e)))?;
            image_bytes = Some(data.to_vec());
        } else if name == "options" {
            let text = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Options read error: {}", e)))?;
            if let Ok(parsed) = serde_json::from_str::<ProcessOptions>(&text) {
                options = parsed;
            }
        }
    }

    let bytes = image_bytes
        .ok_or((StatusCode::BAD_REQUEST, "Missing 'image' file field".to_string()))?;

    let dynamic_img = image::load_from_memory(&bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to decode image: {}", e)))?;

    let output_img = ImageProcessor::process(&dynamic_img, &options);

    let mut png_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut png_bytes);
    DynamicImage::ImageRgba8(output_img)
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to encode PNG: {}", e)))?;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "image/png".parse().unwrap());
    headers.insert(
        header::CONTENT_DISPOSITION,
        "inline; filename=\"cleaned_drawing.png\"".parse().unwrap(),
    );

    Ok((StatusCode::OK, headers, png_bytes).into_response())
}

async fn sample_handler(
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Response, (StatusCode, String)> {
    let (png_bytes, content_type) = samples::get_sample_drawing(&name)
        .ok_or((StatusCode::NOT_FOUND, "Sample not found".to_string()))?;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());

    Ok((StatusCode::OK, headers, png_bytes).into_response())
}

async fn isolate_handler(
    Json(payload): Json<IsolateJsonRequest>,
) -> Result<Response, (StatusCode, String)> {
    let mut child = tokio::process::Command::new("python3")
        .arg("dimension_cleaner.py")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to spawn cleaner: {}", e)))?;

    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let req_json = serde_json::to_vec(&serde_json::json!({
            "image_base64": payload.image_base64
        })).unwrap();
        stdin.write_all(&req_json).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to send data to cleaner: {}", e)))?;
    }

    let output = child.wait_with_output().await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Cleaner execution failed: {}", e)))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Cleaner error: {}", err)));
    }

    let body_str = String::from_utf8(output.stdout)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Invalid cleaner output: {}", e)))?;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    Ok((StatusCode::OK, headers, body_str).into_response())
}


