#![cfg(feature = "prompt-text-hashed")]

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::Tokenizer;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio::task::JoinSet;
use tokio::time::{sleep, timeout};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("request-sim-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_tokenizer(directory: &Path) {
    let model = WordLevel::builder()
        .vocab(
            [("hello", 0), ("world", 1), ("test", 2)]
                .into_iter()
                .map(|(word, id)| (word.to_owned(), id))
                .collect(),
        )
        .unk_token("test".to_owned())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
        .save(directory.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        directory.join("tokenizer_config.json"),
        json!({
            "tokenizer_class": "Qwen2Tokenizer",
            "added_tokens_decoder": {
                "151643": {"content": "<|endoftext|>"},
                "151644": {"content": "<|im_start|>"},
                "151645": {"content": "<|im_end|>"}
            }
        })
        .to_string(),
    )
    .unwrap();
}

async fn read_request(socket: &mut TcpStream) -> Value {
    let mut reader = BufReader::new(socket);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();
    assert_eq!(line.trim_end(), "POST /v1/chat/completions HTTP/1.1");
    let mut content_length = None;
    loop {
        line.clear();
        assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
        let header = line.trim_end();
        if header.is_empty() {
            break;
        }
        let (name, value) = header.split_once(':').unwrap();
        if name.eq_ignore_ascii_case("content-length") {
            content_length = Some(value.trim().parse().unwrap());
        }
    }
    let mut body = vec![0; content_length.unwrap()];
    reader.read_exact(&mut body).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

async fn mock_vllm_response(mut socket: TcpStream) {
    socket.set_nodelay(true).unwrap();
    let request = read_request(&mut socket).await;
    assert_eq!(request["model"], "Qwen/Qwen3-30B-A3B");
    assert_eq!(request["stream"], true);
    assert_eq!(request["stream_options"], json!({"include_usage": true}));
    assert_eq!(request["ignore_eos"], true);
    assert_eq!(
        request["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    assert!(request.get("min_tokens").is_none());

    let max_tokens = request["max_tokens"].as_u64().unwrap();
    if max_tokens == 48 {
        socket
            .write_all(
                b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        return;
    }

    // The 200-token trace target is capped to 128 - 1 input token - 32 template tokens.
    let with_usage = max_tokens == 95;
    assert!(
        with_usage || max_tokens == 64,
        "unexpected max_tokens {max_tokens}"
    );
    if with_usage {
        sleep(Duration::from_millis(100)).await;
    }
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        )
        .await
        .unwrap();
    socket
        .write_all(
            concat!(
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":null}}]}\n\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    if with_usage {
        sleep(Duration::from_millis(50)).await;
    }
    let reasoning = if with_usage {
        // The role field is legal on a chunk that also has generated text.
        r#"data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"think"}}]}"#
    } else {
        r#"data:{"choices":[{"delta":{"reasoning":"think"}}]}"#
    };
    socket.write_all(reasoning.as_bytes()).await.unwrap();
    socket.write_all(b"\n\n").await.unwrap();
    if with_usage {
        sleep(Duration::from_millis(150)).await;
    }
    socket
        .write_all(
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    if with_usage {
        socket
            .write_all(
                concat!(
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":18,\"completion_tokens\":7,",
                "\"total_tokens\":25,\"prompt_tokens_details\":{\"cached_tokens\":16}}}\n\n"
            )
                .as_bytes(),
            )
            .await
            .unwrap();
    }
    socket.write_all(b"data: [DONE]\n\n").await.unwrap();
}

fn number(metrics: &Value, field: &str) -> f64 {
    metrics[field].as_str().unwrap().parse().unwrap()
}

#[tokio::test]
async fn replays_milliseconds_and_reports_openai_usage_and_send_to_token_ttft() {
    let directory = TestDirectory::new();
    write_tokenizer(&directory.0);
    std::fs::write(
        directory.0.join("trace.jsonl"),
        concat!(
            "{\"timestamp\":0,\"input_length\":1,\"output_length\":200,\"hash_ids\":[1]}\n",
            "{\"timestamp\":142,\"input_length\":1,\"output_length\":64,\"hash_ids\":[2]}\n",
            "{\"timestamp\":200,\"input_length\":1,\"output_length\":48,\"hash_ids\":[3]}\n",
        ),
    )
    .unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!(
        "http://{}/v1/chat/completions",
        listener.local_addr().unwrap()
    );
    let server = tokio::spawn(async move {
        let mut connections = JoinSet::new();
        for _ in 0..3 {
            let (socket, _) = listener.accept().await.unwrap();
            connections.spawn(mock_vllm_response(socket));
        }
        while let Some(result) = connections.join_next().await {
            result.unwrap();
        }
    });

    let output = timeout(
        Duration::from_secs(10),
        Command::new(env!("CARGO_BIN_EXE_request-sim"))
            .current_dir(&directory.0)
            .kill_on_drop(true)
            .args([
                "--api",
                "openai",
                "--endpoint",
                &endpoint,
                "--model-name",
                "Qwen/Qwen3-30B-A3B",
                "--dataset",
                "mooncake",
                "--dataset-path",
                "trace.jsonl",
                "--tokenizer",
                "tokenizer.json",
                "--tokenizer-config",
                "tokenizer_config.json",
                "--channel-capacity",
                "4",
                "--scale-factor",
                "1.0",
                "--context-length",
                "128",
                "--ignore-eos",
                "--disable-thinking",
                "--stream",
                "--threads",
                "2",
                "--output-path",
                "output.jsonl",
                "--summary-path",
                "summary.json",
            ])
            .output(),
    )
    .await
    .expect("trace replay should finish within 10 seconds")
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    timeout(Duration::from_secs(1), server)
        .await
        .unwrap()
        .unwrap();

    let jsonl = std::fs::read_to_string(directory.0.join("output.jsonl")).unwrap();
    let mut requests: Vec<Value> = jsonl
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(requests.len(), 3, "{jsonl}");
    requests.sort_by(|left, right| number(left, "s_time").total_cmp(&number(right, "s_time")));
    let [exact, fallback, rejected] = requests.as_slice() else {
        unreachable!()
    };

    assert_eq!(exact["output_length"], "200");
    // 100 ms before headers plus 50 ms before the reasoning delta.
    assert!(number(exact, "first_token_time") >= 150.0, "{exact}");
    assert!(number(exact, "total_time") >= 150.0, "{exact}");
    assert_eq!(exact["token_count"], "2");
    assert_eq!(exact["actual_output_tokens"], "7");
    assert_eq!(exact["output_tokens_source"], "usage");
    assert_eq!(exact["prompt_tokens"], "18");
    assert_eq!(exact["cached_tokens"], "16");
    assert_eq!(exact["output_text"], "answer");

    assert!(number(fallback, "s_time") >= 142.0, "{fallback}");
    assert_eq!(fallback["output_length"], "64");
    assert_eq!(fallback["token_count"], "2");
    assert_eq!(fallback["actual_output_tokens"], "2");
    assert_eq!(fallback["output_tokens_source"], "sse_delta");
    assert!(fallback.get("cached_tokens").is_none());
    assert_eq!(rejected["status"], "400");

    let summary: Value =
        serde_json::from_slice(&std::fs::read(directory.0.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["requests_total"], "3");
    assert_eq!(summary["requests_success"], "2");
    assert_eq!(summary["output_tokens_total"], "9");
    let expected_tps = 9_000.0 / number(&summary, "duration_ms");
    assert!((number(&summary, "throughput_tps") - expected_tps).abs() < 0.001);
}
