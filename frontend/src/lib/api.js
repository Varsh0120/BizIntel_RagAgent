import { supabase } from "./supabase";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

async function getAccessToken() {
  const {
    data: { session },
    error,
  } = await supabase.auth.getSession();

  if (error) {
    throw error;
  }

  if (!session?.access_token) {
    throw new Error("Please log in first.");
  }

  return session.access_token;
}

async function request(path, options = {}) {
  const accessToken = await getAccessToken();

  const headers = new Headers(options.headers || {});

  headers.set(
    "Authorization",
    `Bearer ${accessToken}`
  );

  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      ...options,
      headers,
    }
  );

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : { detail: await response.text() };

  if (!response.ok) {
    throw new Error(
      body?.detail ||
        `Request failed with status ${response.status}`
    );
  }

  return body;
}

export async function askQuestion(payload) {
  return request("/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function uploadDocument(file) {
  const formData = new FormData();

  formData.append("file", file);

  return request("/upload-document", {
    method: "POST",
    body: formData,
  });
}

export async function listDocuments() {
  return request("/documents");
}

export async function deleteDocument(documentId) {
  return request(`/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
}

export async function listConversations() {
  return request("/conversations");
}

export async function getConversationMessages(conversationId) {
  return request(
    `/conversations/${encodeURIComponent(conversationId)}/messages`
  );
}
