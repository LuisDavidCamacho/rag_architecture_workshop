const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

export type ChatRole = "user" | "assistant";

export interface QueryPayload {
  query: string;
}

export interface QueryResponse {
  chat_id: string;
  response: string;
}

export interface EmbedResponse {
  embedded_documents: number;
  message: string;
}

async function request<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    let detail = await response.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail ?? parsed.message ?? detail;
    } catch {
      // ignore
    }
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function startChat(payload: QueryPayload): Promise<QueryResponse> {
  return request<QueryResponse>("/query", payload);
}

export function continueChat(
  chatId: string,
  payload: QueryPayload
): Promise<QueryResponse> {
  return request<QueryResponse>(`/query/${chatId}`, payload);
}

export function embedDocuments(
  filename: string,
  options?: { chunkSize?: number; overlap?: number }
): Promise<EmbedResponse> {
  return request<EmbedResponse>("/embed", {
    filename,
    chunk_size: options?.chunkSize,
    overlap: options?.overlap
  });
}
