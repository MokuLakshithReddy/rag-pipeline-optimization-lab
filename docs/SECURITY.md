# Security & Reliability Engineering Standards

## 1. Threat Model & Mitigations

| Threat Vector | Attack Mechanism | Engineering Mitigation |
| :--- | :--- | :--- |
| **Path Traversal** | Submitting `../../etc/passwd` or `..\boot.ini` in filenames | `sanitize_filename()` strips all path separators, control characters, and leading dots. `validate_file_path()` verifies that resolved target stays within base directory. |
| **Resource Exhaustion** | Uploading gigabyte files to crash memory | Strict 25MB maximum file size limit enforced prior to stream parsing (`validate_file_metadata`). |
| **Malicious Code Execution** | Uploading `.exe`, `.sh`, `.bat` disguised as text | Whitelisting allowed extensions (`.pdf`, `.txt`, `.md`, `.json`, `.csv`). Binary null-byte detection rejects binary files. |
| **Indirect Prompt Injection** | Embedding malicious jailbreak instructions in documents | Context is encapsulated in `<untrusted_context>` tags. System instructions strictly forbid the LLM from executing commands inside context. |
| **Information Disclosure** | Stack traces or credentials leaked in API errors | Global exception middleware catches errors and returns sanitized JSON with request tracking ID. Internal stack traces are logged exclusively to server logs. |
| **CORS Exploitation** | Cross-origin credential theft | Configurable origin whitelisting (`ALLOWED_ORIGINS`). |
| **Clickjacking & MIME-Sniffing** | Framing or MIME-type override | Security response headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`. |

