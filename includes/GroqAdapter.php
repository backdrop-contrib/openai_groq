<?php
/**
 * Adapter for the Groq provider (OpenAI-compatible API).
 *
 * See https://console.groq.com/docs/api for details.
 */

// Prefer Composer Manager's autoloader when available; fall back to module vendor.
$__openai_sdk_source =& backdrop_static('openai_sdk_source');
if (module_exists('composer_manager')) {
  if (function_exists('composer_manager_register_autoloader')) {
    composer_manager_register_autoloader();
  }
  $__openai_sdk_source = 'composer_manager';
}
else {
  $autoload = BACKDROP_ROOT . '/' . backdrop_get_path('module', 'openai') . '/vendor/autoload.php';
  if (file_exists($autoload)) {
    require_once $autoload;
    $__openai_sdk_source = 'module_vendor';
  }
  else {
    $__openai_sdk_source = $__openai_sdk_source ?: 'unknown';
  }
}

use OpenAI\Client as OpenAIClient;
use OpenAI\Exceptions\TransporterException;

/**
 * Avoid depending on Symfony's StreamedResponse; use minimal anonymous
 * streaming objects with a send() method to keep adapters self-contained.
 */

class GroqAdapter implements AIClientInterface {
  protected $client;
  protected $api;
  protected $realApiKey;
  protected $provider_id = 'groq';

  public function __construct($api_key = NULL, ?OpenAIApi $api = NULL) {
    $this->realApiKey = trim((string) ($api_key ?? ''));
    $this->api = $api;

    // Groq exposes an OpenAI-compatible API. Use a dummy key for SDK
    // validation and set headers so the real key is used in Authorization.
    $dummyKey = 'sk-groq-dummy-key-0000000000000000000000';

    // Use the documented Groq base (console.groq / api.groq). The SDK will
    // request /v1/* paths so include /v1 in the base URI.
    $baseUri = 'https://api.groq.com/v1';

    $factory = \OpenAI::factory()
      ->withApiKey($dummyKey)
      ->withBaseUri($baseUri)
      ->make();

    // The OpenAI PHP SDK doesn't accept a second header override in factory
    // in all versions; we will keep the real key in $this->realApiKey and
    // attach it to requests where we use curl directly. For SDK calls we
    // rely on the dummy key where possible; many Groq endpoints accept the
    // API key in the Authorization header and the SDK may be used with a
    // dummy key only. We still keep the factory client for convenience.
    $this->client = $factory;
  }

  // Model discovery - return associative arrays of model_id => display name.
  public function getModels(): array {
    $models = [];
    try {
      // Try SDK list() call first. Use toArray() when available.
      $list = NULL;
      try {
        $list = $this->client->models()->list();
        if (method_exists($list, 'toArray')) {
          $list = $list->toArray();
        }
      }
      catch (\Exception $e) {
        // SDK call failed; fall back to static set below.
        $list = NULL;
      }

      if (!empty($list) && is_array($list) && !empty($list['data'])) {
        foreach ($list['data'] as $model) {
          $id = $model['id'] ?? ($model['name'] ?? NULL);
          if (empty($id)) continue;
          $label = $model['name'] ?? $id;
          $models[$id] = $label;
        }
      }
    }
    catch (TransporterException | \Exception $e) {
      watchdog('openai_groq', 'Failed to fetch Groq models via SDK: @error', ['@error' => $e->getMessage()], WATCHDOG_DEBUG);
    }

    // If SDK didn't return models, try Groq's documented OpenAI-compatible
    // models endpoint: GET https://api.groq.com/openai/v1/models
    if (empty($models)) {
      try {
        $url = 'https://api.groq.com/openai/v1/models';
        $ch = curl_init($url);
        curl_setopt_array($ch, [
          CURLOPT_RETURNTRANSFER => TRUE,
          CURLOPT_HTTPHEADER => [
            'Authorization: Bearer ' . $this->realApiKey,
            'Accept: application/json',
            'Referer: ' . url('<front>', ['absolute' => TRUE]),
          ],
          CURLOPT_TIMEOUT => 10,
        ]);
        $resp = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        if ($http_code === 200 && !empty($resp)) {
          $data = json_decode($resp, TRUE);
          if (!empty($data['data']) && is_array($data['data'])) {
            foreach ($data['data'] as $m) {
              $id = $m['id'] ?? ($m['name'] ?? NULL);
              if (empty($id)) continue;
              $label = $m['name'] ?? $id;
              $models[$id] = $label;
            }
          }
        }
      }
      catch (\Exception $e) {
        watchdog('openai_groq', 'Failed to fetch Groq models via HTTP fallback: @error', ['@error' => $e->getMessage()], WATCHDOG_DEBUG);
      }
    }

    // If we didn't get any models from the API, fall back to a curated list.
    if (empty($models)) {
      $models = [
        'llama3-70b-8192' => 'Llama 3 70B (8K)',
        'mixtral-8x7b-32768' => 'Mixtral 8x7B (32K)',
        'gemma-7b-it' => 'Gemma 7B IT',
        'llama2-70b-4096' => 'Llama 2 70B (4K)',
      ];
    }

    asort($models);
    return $models;
  }

  /**
   * Filter models by capability.
   *
   * Groq currently focuses on text/chat models. This helper provides a
   * best-effort capability filter and allows alter hooks for site-specific
   * overrides.
   *
   * @param string $capability
   *   The capability to filter by: 'text', 'vision', 'image', 'embeddings', 'moderation'.
   *
   * @return array
   *   Filtered array of model_id => display name.
   */
  public function getModelsByCapability($capability): array {
    $all_models = $this->getModels();
    $filtered = [];

    foreach ($all_models as $id => $name) {
      $ok = FALSE;
      switch ($capability) {
        case 'text':
          $ok = TRUE; // Groq models are primarily text/chat capable
          break;
        case 'embeddings':
        case 'embedding':
          // Groq does not currently expose embeddings via the OpenAI API
          $ok = FALSE;
          break;
        case 'image':
        case 'vision':
          // No image generation or vision inputs supported today
          $ok = FALSE;
          break;
        case 'moderation':
          $ok = FALSE;
          break;
      }
      if ($ok) {
        $filtered[$id] = $name;
      }
    }

    backdrop_alter('openai_model_capabilities', $filtered, $capability, $this->provider_id);
    return $filtered;
  }

  public function getChatModels(): array { return $this->getModelsByCapability('text'); }
  public function getImageModels(): array { return $this->getModelsByCapability('image'); }
  public function getVisionModels(): array { return $this->getModelsByCapability('vision'); }
  public function getEmbeddingModels(): array { return $this->getModelsByCapability('embeddings'); }
  public function getModerationModels(): array { return $this->getModelsByCapability('moderation'); }

  // ------------------------ Text / Chat ------------------------
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    $start_time = microtime(TRUE);
     try {
       $payload = [
         'model' => $model,
         'prompt' => trim($prompt),
         'temperature' => (float) $temperature,
       ];

       if ((int) $max_tokens > 0) {
         $payload['max_tokens'] = (int) $max_tokens;
       }

       if ($stream_response) {
        // Try SDK streamed path if available
        try {
          $stream = $this->client->completions()->createStreamed($payload);
          return new class($stream) {
            protected $stream;
            public function __construct($stream) { $this->stream = $stream; }
            public function send() {
              foreach ($this->stream as $data) {
                $text = $data->choices[0]->delta->content ?? $data->choices[0]->text ?? '';
                if ($text !== '') { echo $text; @ob_flush(); @flush(); }
              }
            }
          };
         }
         catch (\Throwable $e) {
           // Fall back to non-streaming below
         }
       }

      // Non-streaming path: use SDK if possible, otherwise curl.
      try {
        $resp = $this->client->completions()->create($payload);
        if (method_exists($resp, 'toArray')) {
          $resp = $resp->toArray();
        }
        // Normalized location for text
        $out_text = trim($resp['choices'][0]['text'] ?? $resp['choices'][0]['message']['content'] ?? '');
        return $out_text;
      }
      catch (\Exception $e) {
        // As a last resort, use curl with the real API key.
        $url = 'https://api.groq.com/v1/completions';
        $ch = curl_init($url);
        curl_setopt_array($ch, [
          CURLOPT_RETURNTRANSFER => TRUE,
          CURLOPT_POST => TRUE,
          CURLOPT_POSTFIELDS => json_encode($payload),
          CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'Authorization: Bearer ' . $this->realApiKey,
            'Referer: ' . url('<front>', ['absolute' => TRUE]),
            'X-Title: ' . (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
          ],
        ]);
        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        if ($http_code == 200) {
          $decoded = json_decode($response, TRUE);
          $out_text = trim($decoded['choices'][0]['text'] ?? $decoded['choices'][0]['message']['content'] ?? '');
          return $out_text;
        }
        throw new \Exception('HTTP ' . $http_code . ': ' . substr((string)$response, 0, 500));
      }
    }
    catch (TransporterException | \Exception $e) {
      watchdog('openai_groq', 'Groq completions error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return '';
    }
  }

  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE) {
    $start_time = microtime(TRUE);
     try {
      $payload = [
        'model' => $model,
        'messages' => $messages,
        'temperature' => (float) $temperature,
      ];

      if ((int) $max_tokens > 0) {
        $payload['max_tokens'] = (int) $max_tokens;
      }

      // First try SDK streaming / non-streaming paths as before.
      if ($stream_response) {
        try {
          $stream = $this->client->chat()->createStreamed($payload);
          $streamed = new class($stream) {
            protected $stream;
            public function __construct($stream) { $this->stream = $stream; }
            public function send() {
              foreach ($this->stream as $data) {
                $text = $data->choices[0]->delta->content ?? $data->choices[0]->message->content ?? $data->choices[0]->text ?? '';
                if ($text !== '') { echo $text; @ob_flush(); @flush(); }
              }
            }
          };
         return $streamed;
        }
        catch (\Throwable $e) {
          // Fall through to curl-based fallback below.
        }
      }

      // Try SDK non-streaming
      try {
        $resp = $this->client->chat()->create($payload);
        if (method_exists($resp, 'toArray')) {
          $resp = $resp->toArray();
        }
        $out_text = trim($resp['choices'][0]['message']['content'] ?? $resp['choices'][0]['text'] ?? '');
        return $out_text;
      }
      catch (\Exception $e) {
        // SDK likely cannot reach Groq's OpenAI-incompatible chat endpoint.
        // Fall back to calling Groq's model outputs endpoint directly using the
        // real API key and a prompt-based translation of the messages.
      }

      // --- Fallback: translate messages to a single prompt and call Groq model outputs ---
      // Build simple prompt preserving roles. This is a conservative translation
      // that works even if Groq doesn't support OpenAI chat shapes.
      $prompt = '';
      foreach ($messages as $m) {
        if (empty($m['content'])) {
          continue;
        }
        $role = isset($m['role']) ? strtolower($m['role']) : 'user';
        switch ($role) {
          case 'system':
            $prompt .= $m['content'] . "\n\n";
            break;
          case 'assistant':
            $prompt .= "Assistant: " . $m['content'] . "\n";
            break;
          case 'user':
          default:
            $prompt .= "User: " . $m['content'] . "\n";
            break;
        }
      }
      $prompt .= "Assistant: ";

      // Build request body; note: Groq's documented endpoint uses POST /v1/outputs
      // with a 'model' key in the body. We'll try a sequence of candidate endpoints
      // so the adapter works across versions.
      $body = [
        'model' => $model,
        'input' => $prompt,
        'temperature' => (float) $temperature,
      ];
      if ((int) $max_tokens > 0) {
        $body['max_output_tokens'] = (int) $max_tokens;
        $body['max_tokens'] = (int) $max_tokens;
      }

      // Chat-shaped payload (used when hitting chat/completions endpoints)
      $chat_payload = [
        'model' => $model,
        'messages' => $messages,
        'temperature' => (float) $temperature,
      ];
      if ((int) $max_tokens > 0) {
        $chat_payload['max_tokens'] = (int) $max_tokens;
      }

      // Normalize base host: allow both https://api.groq.com and https://api.groq.com/v1
      $base_with_v1 = 'https://api.groq.com/v1';
      $root_host = preg_replace('|/v1/?$|', '', $base_with_v1);

      // Candidate endpoints in preferred order. Try Groq's OpenAI-compatible
      // prefix (/openai/v1/*) first (chat path works per their docs), then
      // try other variants and legacy paths.
      $candidates = [
        // Groq OpenAI-compatible chat path documented in their API reference
        $root_host . '/openai/v1/chat/completions',
        // Groq outputs/completions variants
        $root_host . '/openai/v1/outputs',
        $root_host . '/openai/v1/completions',
        // fallback to /v1 and root paths
        $root_host . '/v1/outputs',
        $root_host . '/outputs',
        $root_host . '/v1/completions',
        $root_host . '/completions',
        // Model-scoped fallbacks (prefer OpenAI-compatible path)
        $root_host . '/openai/v1/models/' . rawurlencode($model) . '/outputs',
        $root_host . '/v1/models/' . rawurlencode($model) . '/outputs',
      ];

      // Helper to attempt a non-streaming POST and return decoded response or NULL on failure.
      $tryPost = function ($url, $body) {
        $ch = curl_init($url);
        curl_setopt_array($ch, [
          CURLOPT_RETURNTRANSFER => TRUE,
          CURLOPT_POST => TRUE,
          CURLOPT_POSTFIELDS => json_encode($body),
          CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'Authorization: Bearer ' . $this->realApiKey,
            'Referer: ' . url('<front>', ['absolute' => TRUE]),
            'X-Title: ' . (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
          ],
          CURLOPT_TIMEOUT => 30,
        ]);
        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        if ($http_code === 200 || $http_code === 201) {
          $decoded = json_decode($response, TRUE);
          watchdog('openai_groq', 'Groq candidate POST to @url succeeded HTTP @code', ['@url' => $url, '@code' => $http_code], WATCHDOG_DEBUG);
          return $decoded ?? $response;
        }

        // Log non-200 responses for debugging, but trim body length.
        $trimmed = substr((string)$response, 0, 500);
        watchdog('openai_groq', 'Groq candidate POST to @url returned HTTP @code: @body', ['@url' => $url, '@code' => $http_code, '@body' => $trimmed], WATCHDOG_WARNING);
        return NULL;
      };

      // STREAMING path: try candidates until one returns 200+ stream behavior
      if ($stream_response) {
        foreach ($candidates as $url) {
          try {
            // Choose the appropriate payload for chat vs prompt endpoints
            $request_payload = (stripos($url, '/chat/') !== FALSE || stripos($url, '/chat.completions') !== FALSE) ? $chat_payload : $body;

            return new class($url, $request_payload) {
              protected $url;
              protected $request_payload;
              public function __construct($url, $request_payload) {
                $this->url = $url;
                $this->request_payload = $request_payload;
              }
              public function send() {
                $ch = curl_init($this->url);
                $payload = json_encode($this->request_payload);
                curl_setopt($ch, CURLOPT_POST, TRUE);
                curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
                curl_setopt($ch, CURLOPT_RETURNTRANSFER, FALSE);
                curl_setopt($ch, CURLOPT_WRITEFUNCTION, function ($ch, $data) {
                  $parts = preg_split('/\r?\n\r?\n/', $data);
                  foreach ($parts as $part) {
                    $part = trim($part);
                    if ($part === '') { continue; }
                    $part = preg_replace('/^data:\s*/m', '', $part);
                    $decoded = json_decode($part, TRUE);
                    $sent = '';
                    if (is_array($decoded)) {
                      if (!empty($decoded['outputs']) && is_array($decoded['outputs'])) {
                        $out = $decoded['outputs'][0] ?? NULL;
                        if (is_string($out)) { $sent = $out; }
                        elseif (is_array($out)) { $sent = $out['content'] ?? $out['text'] ?? $out['output'] ?? json_encode($out); }
                      }
                      elseif (!empty($decoded['choices']) && is_array($decoded['choices'])) {
                        $choice = $decoded['choices'][0] ?? [];
                        $sent = $choice['delta']['content'] ?? $choice['message']['content'] ?? $choice['text'] ?? '';
                      }
                      elseif (!empty($decoded['data']) && is_array($decoded['data'])) {
                        $d = $decoded['data'][0] ?? [];
                        if (is_array($d)) { $sent = $d['content'] ?? $d['text'] ?? ''; }
                      }
                    }
                    else { $sent = $part; }

                    if ($sent !== '') { echo $sent; @ob_flush(); @flush(); }
                  }
                  return strlen($data);
                });
                curl_setopt($ch, CURLOPT_HTTPHEADER, [
                  'Content-Type: application/json',
                  'Accept: text/event-stream, application/json',
                  'Authorization: Bearer ' . $this->realApiKey,
                  'Referer: ' . url('<front>', ['absolute' => TRUE]),
                  'X-Title: ' . (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
                ]);
                curl_setopt($ch, CURLOPT_TIMEOUT, 0);
                curl_setopt($ch, CURLOPT_BUFFERSIZE, 1024);
                $ok = @curl_exec($ch);
                if ($ok === FALSE) {
                  $err = curl_error($ch);
                  curl_close($ch);
                  watchdog('openai_groq', 'Groq streaming curl error on @url: @error', ['@url' => $this->url, '@error' => $err], WATCHDOG_WARNING);
                  echo '';
                }
                else { curl_close($ch); }
              }
            };
          }
          catch (\Exception $e) {
            // Try next candidate
            watchdog('openai_groq', 'Streaming candidate failed @url: @msg', ['@url' => $url, '@msg' => $e->getMessage()], WATCHDOG_DEBUG);
            continue;
          }
         }
         // If we reached here, streaming failed for all candidates; fall through to non-streaming
       }

      // Non-streaming: try candidates in order and return the first successful normalized text
      foreach ($candidates as $url) {
        $use_body = (stripos($url, '/chat/') !== FALSE || stripos($url, '/chat.completions') !== FALSE) ? $chat_payload : $body;
        $decoded = $tryPost($url, $use_body);
        if (!empty($decoded)) {
          // Normalize shapes
          if (!empty($decoded['outputs']) && is_array($decoded['outputs'])) {
            $out = $decoded['outputs'][0];
            if (is_string($out)) { return trim($out); }
            if (is_array($out)) {
              if (!empty($out['content'])) { return trim(is_string($out['content']) ? $out['content'] : json_encode($out['content'])); }
              if (!empty($out['text'])) { return trim($out['text']); }
              if (!empty($out['output'])) { return trim(is_string($out['output']) ? $out['output'] : json_encode($out['output'])); }
              return trim(json_encode($out));
            }
          }
          if (!empty($decoded['choices'][0]['message']['content'])) { return trim($decoded['choices'][0]['message']['content']); }
          if (!empty($decoded['choices'][0]['text'])) { return trim($decoded['choices'][0]['text']); }
          // If it's raw string-like response
          if (is_string($decoded) && trim($decoded) !== '') { return trim($decoded); }
        }

        // If a candidate failed, also try swapping 'input' -> 'inputs' shape and retry once
        $alt_body = $use_body;
        if (isset($alt_body['input'])) {
          $alt_body['inputs'] = [$alt_body['input']];
          unset($alt_body['input']);
          $decoded2 = $tryPost($url, $alt_body);
          if (!empty($decoded2)) {
            if (!empty($decoded2['outputs']) && is_array($decoded2['outputs'])) {
              $out = $decoded2['outputs'][0];
              if (is_string($out)) { return trim($out); }
              if (is_array($out)) {
                if (!empty($out['content'])) { return trim(is_string($out['content']) ? $out['content'] : json_encode($out['content'])); }
                if (!empty($out['text'])) { return trim($out['text']); }
                if (!empty($out['output'])) { return trim(is_string($out['output']) ? $out['output'] : json_encode($out['output'])); }
                return trim(json_encode($out));
              }
            }
            if (!empty($decoded2['choices'][0]['message']['content'])) { return trim($decoded2['choices'][0]['message']['content']); }
            if (!empty($decoded2['choices'][0]['text'])) { return trim($decoded2['choices'][0]['text']); }
            if (is_string($decoded2) && trim($decoded2) !== '') { return trim($decoded2); }
          }
        }
      }

      // If nothing succeeded, log and return empty string
      watchdog('openai_groq', 'Groq chat HTTP error: no endpoint succeeded for model @model', ['@model' => $model], WATCHDOG_WARNING);
      return '';
    }
    catch (TransporterException | \Exception $e) {
      watchdog('openai_groq', 'Groq completions error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return '';
    }
  }

  // ------------------------ Images / Moderation / Embeddings ------------------------
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    // Groq does not currently support image generation via the OpenAI-compatible API
    return ['data' => []];
  }

  public function moderation(string $input, string $model = 'omni-moderation-latest'): array {
    // Not supported by Groq at this time
    return [];
  }

  public function embed(string $model, $input) {
    // Backwards-compatible alias for embedding()
    return $this->embedding((string)$input, $model);
  }

  public function embedding(string $input, string $model, bool $log = TRUE): array {
    $start_time = microtime(TRUE);
    try {
      // Attempt SDK path
      $resp = $this->client->embeddings()->create([
        'model' => $model,
        'input' => $input,
      ]);
      $result = [];
      if (method_exists($resp, 'toArray')) {
        $result = $resp->toArray();
      }
      $vector = $result['data'][0]['embedding'] ?? [];
      if (isset($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], $result, TRUE, $duration, NULL, !$log);
      }
      return $vector;
    }
    catch (\Exception $e) {
      if (isset($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], NULL, FALSE, $duration, $e->getMessage(), !$log);
      }
      if ($log) {
        $error_msg = $e->getMessage();
        // Suppress log if it's a "does not support embeddings" or similar during probing.
        if (strpos($error_msg, 'does not support embeddings') === FALSE && strpos($error_msg, 'not found') === FALSE) {
          watchdog('openai_groq', 'Groq embedding error: @error', ['@error' => $error_msg], WATCHDOG_WARNING);
        }
      }
      return [];
    }
  }

  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    throw new \Exception('Text-to-speech is not supported by Groq');
  }

  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    throw new \Exception('Speech-to-text is not supported by Groq');
  }
}
