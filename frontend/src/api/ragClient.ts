const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

export interface QueryPayload {
  query: string;
}

export interface QueryResponse {
  chat_id: string;
  response: string;
}

export interface EmbedRequest {
  documents: string[];
  chunk_size?: number;
  overlap?: number;
}

export interface EmbedResponse {
  embedded_documents: number;
  message: string;
}

async function request<T>(path: string, payload: QueryPayload): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function startChat(payload: QueryPayload): Promise<QueryResponse> {
  return request<QueryResponse>("/query", payload);
}

export async function continueChat(
  chatId: string,
  payload: QueryPayload
): Promise<QueryResponse> {
  return request<QueryResponse>(`/query/${chatId}`, payload);
}

export async function embedDocuments(
  payload: EmbedRequest
): Promise<EmbedResponse> {
  const response = await fetch(`${API_BASE_URL}/embed`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Embedding failed with status ${response.status}`);
  }

  return (await response.json()) as EmbedResponse;
}
