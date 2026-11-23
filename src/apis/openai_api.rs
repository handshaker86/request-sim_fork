use super::{
    compute_content_hash_rid, InFlightState, LLMApi, RequestError, RidSource, CONTEXT_LENGTH,
    DISABLE_THINKING, IGNORE_EOS, MAX_TOKENS_CAP, METRIC_PERCENTILES, MODEL_NAME, RID_SOURCE,
};
use crate::dataset::PromptPayload;
use futures_util::TryStreamExt;
use reqwest::Response;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    time::{timeout_at, Instant},
};
use tokio_util::io::StreamReader;

#[derive(Copy, Clone)]
pub struct OaiApi;

const DEFAULT_PERCENTILES: [u32; 3] = [90, 95, 99];
// Reserve room for Qwen chat role markers and the assistant generation prefix.
const CHAT_TEMPLATE_MARGIN: u64 = 32;

fn record_usage(result: &mut BTreeMap<String, String>, response: &Value) {
    for (path, metric) in [
        ("/usage/prompt_tokens", "prompt_tokens"),
        ("/usage/completion_tokens", "actual_output_tokens"),
        (
            "/usage/prompt_tokens_details/cached_tokens",
            "cached_tokens",
        ),
    ] {
        if let Some(count) = response.pointer(path).and_then(Value::as_u64) {
            result.insert(metric.to_string(), count.to_string());
        }
    }
    if response
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .is_some()
    {
        result.insert("output_tokens_source".to_string(), "usage".to_string());
    }
}

#[async_trait::async_trait]
impl LLMApi for OaiApi {
    fn request_json_body(
        prompt: PromptPayload,
        input_length: u64,
        output_length: u64,
        stream: bool,
    ) -> String {
        let messages = match prompt {
            PromptPayload::Content(text) => json!([{"role": "user", "content": text}]),
            PromptPayload::Messages(val) => val,
            PromptPayload::Body(_) => panic!("Body payload not supported for OpenAI API"),
        };
        let mut body = json!({
            "model": MODEL_NAME.get().unwrap().as_str(),
            "messages": messages,
            "stream": stream,
        });
        if stream {
            body["stream_options"] = json!({"include_usage": true});
        }
        let ignore_eos = IGNORE_EOS.get() == Some(&true);
        if ignore_eos {
            body["ignore_eos"] = json!(true);
        }
        if DISABLE_THINKING.get() == Some(&true) {
            body["chat_template_kwargs"] = json!({"enable_thinking": false});
        }

        let context_length = CONTEXT_LENGTH.get().copied().flatten();
        let requested_max = if output_length > 0 {
            Some(output_length)
        } else {
            MAX_TOKENS_CAP.get().copied().flatten()
        };
        if let Some(mut max_tokens) = requested_max {
            if let Some(context_length) = context_length {
                let available = context_length
                    .saturating_sub(input_length)
                    .saturating_sub(CHAT_TEMPLATE_MARGIN)
                    .max(1);
                max_tokens = max_tokens.min(available);
            } else if output_length > 0 && !ignore_eos {
                body["min_tokens"] = json!(output_length);
            }
            body["max_tokens"] = json!(max_tokens);
        }

        if RID_SOURCE.get() == Some(&RidSource::ContentHash) {
            body["rid"] = json!(compute_content_hash_rid(&body["messages"]));
        }

        body.to_string()
    }

    async fn parse_response(
        response: Response,
        request_start: Instant,
        stream: bool,
        timeout_duration: Duration,
        in_flight: Option<Arc<InFlightState>>,
    ) -> Result<BTreeMap<String, String>, RequestError> {
        let mut result = BTreeMap::new();
        result.insert("status".to_string(), response.status().as_str().to_string());

        if !stream {
            if response.status().is_success() {
                if let Ok(body) = response.text().await {
                    if let Ok(v) = serde_json::from_str::<Value>(&body) {
                        record_usage(&mut result, &v);
                        // Extract output text from non-streaming response
                        if let Some(content) = v
                            .get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("message"))
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            result.insert("output_text".to_string(), content.to_string());
                        }
                    }
                }
            }
            return Ok(result);
        }

        if !response.status().is_success() {
            return Ok(result);
        }

        let stream = response.bytes_stream();
        let stream_reader = StreamReader::new(
            stream.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)),
        );
        let mut reader = BufReader::new(stream_reader);
        let mut line = String::new();
        let mut first_token_time = None;
        let mut last_token_time = None;
        let mut token_count = 0;
        let mut tbt_values: Vec<f64> = Vec::new();
        let mut tbt_except_first: Vec<f64> = Vec::new();
        let deadline = request_start + timeout_duration;

        // Accumulate generated text for output tracking
        let mut output_text = String::new();

        loop {
            line.clear();
            let bytes_read = timeout_at(deadline, reader.read_line(&mut line))
                .await
                .map_err(|_| RequestError::Timeout)?
                .map_err(RequestError::StreamErr)?;
            if bytes_read == 0 {
                break;
            }
            let Some(data) = line.trim().strip_prefix("data:").map(str::trim_start) else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }

            let received_at = Instant::now();
            let Ok(chunk) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(error) = chunk.get("error").filter(|error| !error.is_null()) {
                result.insert("stream_error".to_string(), error.to_string());
                break;
            }
            if chunk.get("usage").is_some_and(Value::is_object) {
                record_usage(&mut result, &chunk);
            }

            let Some(delta) = chunk.pointer("/choices/0/delta") else {
                continue;
            };
            let content = delta.get("content").and_then(Value::as_str).unwrap_or("");
            let has_reasoning = ["reasoning_content", "reasoning"].iter().any(|field| {
                delta
                    .get(field)
                    .and_then(Value::as_str)
                    .is_some_and(|text| !text.is_empty())
            });
            if content.is_empty() && !has_reasoning {
                continue;
            }
            output_text.push_str(content);
            token_count += 1;

            if let Some(state) = &in_flight {
                state.output_tokens_so_far.fetch_add(1, Ordering::Relaxed);
                if token_count == 1 {
                    let timestamp = crate::get_timestamp() as u64;
                    state
                        .first_token_time_ms
                        .store(timestamp, Ordering::Release);
                }
            }

            if let Some(last) = last_token_time {
                let tbt = received_at.duration_since(last).as_secs_f64() * 1000.0;
                tbt_values.push(tbt);
                if token_count > 2 {
                    tbt_except_first.push(tbt);
                }
            } else {
                first_token_time = Some(received_at);
                let ttft_ms = received_at.duration_since(request_start).as_secs_f64() * 1000.0;
                result.insert("first_token_time".to_string(), format!("{ttft_ms:.3}"));
            }
            last_token_time = Some(received_at);
        }

        if let Some((first, last)) = first_token_time.zip(last_token_time) {
            let total_time = last.duration_since(first).as_secs_f64() * 1000.0;
            result.insert("total_time".to_string(), format!("{total_time:.3}"));
        }
        if !result.contains_key("actual_output_tokens") {
            result.insert("actual_output_tokens".to_string(), token_count.to_string());
            result.insert("output_tokens_source".to_string(), "sse_delta".to_string());
        }
        if let Some(state) = &in_flight {
            let actual_output_tokens = result["actual_output_tokens"]
                .parse()
                .unwrap_or(token_count);
            state
                .output_tokens_so_far
                .store(actual_output_tokens, Ordering::Relaxed);
        }
        result.insert("token_count".to_string(), token_count.to_string());

        if !output_text.is_empty() {
            result.insert("output_text".to_string(), output_text);
        }

        if !tbt_except_first.is_empty() {
            let max_tbt_except_first = tbt_except_first.iter().copied().fold(f64::MIN, f64::max);
            result.insert(
                "max_time_between_tokens_except_first".to_string(),
                format!("{max_tbt_except_first:.3}"),
            );
        }

        if !tbt_values.is_empty() {
            let max_tbt = tbt_values.iter().copied().fold(f64::MIN, f64::max);
            result.insert(
                "max_time_between_tokens".to_string(),
                format!("{max_tbt:.3}"),
            );
        }

        if !tbt_values.is_empty() {
            let avg_tbt = tbt_values.iter().sum::<f64>() / tbt_values.len() as f64;
            result.insert(
                "avg_time_between_tokens".to_string(),
                format!("{avg_tbt:.3}"),
            );
        }

        // percentile_time_between_tokens
        // need to sort for computing percentage
        if !tbt_values.is_empty() {
            let mut sorted_tbt = tbt_values.clone();
            sorted_tbt
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let len = sorted_tbt.len();
            if len > 0 {
                let percentiles = METRIC_PERCENTILES
                    .get()
                    .map(|v| v.as_slice())
                    .unwrap_or(&DEFAULT_PERCENTILES);
                for percentile in percentiles {
                    let idx = (len as f64 * (*percentile as f64 / 100.0)).ceil() as isize - 1;
                    let idx = idx.max(0) as usize;
                    let idx = idx.min(len - 1);
                    result.insert(
                        format!("p{percentile}_time_between_tokens"),
                        format!("{:.3}", sorted_tbt[idx]),
                    );
                }
            }
        }

        Ok(result)
    }
}
