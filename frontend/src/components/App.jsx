import React, { useEffect, useRef, useState } from "react";
import { supabase } from "../lib/supabase";
import { askQuestion, deleteDocument, getConversationMessages, listConversations, listDocuments, uploadDocument } from "../lib/api";

const ACCEPTED_FILES = ".txt,.md,.markdown,.csv,.json,.pdf,.docx,.xlsx,.xls,.pptx,.ppt,.doc,.html,.htm,.xml,.rtf";
function Icon({ children }) { return <span className="icon" aria-hidden="true">{children}</span>; }

function App() {
  const [session, setSession] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [authMode, setAuthMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loginError, setLoginError] = useState("");
  const [authMessage, setAuthMessage] = useState("");
  const [loggingIn, setLoggingIn] = useState(false);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [conversations, setConversations] = useState([]);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    supabase.auth.getSession().then(({ data, error: sessionError }) => {
      if (!mounted) return;
      if (sessionError) setLoginError(sessionError.message);
      setSession(data.session); setAuthLoading(false);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession); setAuthLoading(false);
    });
    return () => { mounted = false; subscription.unsubscribe(); };
  }, []);

  useEffect(() => { if (session) refreshWorkspace(); }, [session]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, asking]);

  async function refreshWorkspace() {
    setError("");
    const [documentResult, conversationResult] = await Promise.allSettled([listDocuments(), listConversations()]);
    if (documentResult.status === "fulfilled") {
      const nextDocuments = documentResult.value.documents || [];
      setDocuments(nextDocuments);
      setSelectedDocumentId((current) => nextDocuments.some((item) => item.id === current) ? current : (nextDocuments[0]?.id || ""));
    }
    if (conversationResult.status === "fulfilled") {
      setConversations(conversationResult.value.conversations || []);
    }
    const failed = [documentResult, conversationResult].find((result) => result.status === "rejected");
    if (failed) setError(failed.reason?.message || "Workspace refresh failed.");
  }

  async function submitAuth(event) {
    event.preventDefault();
    if (!email.trim() || !password) return setLoginError("Please enter your email and password.");
    setLoggingIn(true); setLoginError(""); setAuthMessage("");
    const credentials = { email: email.trim(), password };
    const { data, error: authError } = authMode === "register" ? await supabase.auth.signUp(credentials) : await supabase.auth.signInWithPassword(credentials);
    if (authError) setLoginError(authError.message);
    else if (data.session) { setSession(data.session); setPassword(""); }
    else { setAuthMessage("Check your email to confirm your account, then sign in."); setAuthMode("login"); setPassword(""); }
    setLoggingIn(false);
  }

  async function recoverPassword() {
    if (!email.trim()) return setLoginError("Enter your email address first.");
    setLoginError("");
    const { error: resetError } = await supabase.auth.resetPasswordForEmail(email.trim(), { redirectTo: window.location.origin });
    if (resetError) setLoginError(resetError.message);
    else setAuthMessage("Password recovery instructions were sent to your email.");
  }

  async function logout() {
    const { error: logoutError } = await supabase.auth.signOut();
    if (logoutError) return setError(logoutError.message);
    setSession(null); setMessages([]); setDocuments([]); setConversations([]); setQuestion(""); setSessionId("");
  }

  async function handleFileSelect(event) {
    const file = event.target.files?.[0]; event.target.value = "";
    if (!file) return;
    setUploading(true); setError("");
    try {
      const data = await uploadDocument(file);
      await refreshWorkspace(); setSelectedDocumentId(data.document_id);
      setMessages((current) => [...current, { id: `upload-${data.document_id}`, role: "assistant", content: `Document “${data.uploaded_file_name}” is ready. ${data.chunks_stored} chunks were indexed.` }]);
    } catch (requestError) { setError(requestError.message); }
    finally { setUploading(false); }
  }

  async function removeDocument(documentId) {
    try { await deleteDocument(documentId); await refreshWorkspace(); }
    catch (requestError) { setError(requestError.message); }
  }

  async function openConversation(conversation) {
    try {
      const data = await getConversationMessages(conversation.id);
      setSessionId(data.conversation.session_id);
      const sourceDocumentId = (data.messages || [])
        .flatMap((message) => message.sources || [])
        .find((source) => source.document_id)?.document_id;
      if (sourceDocumentId && documents.some((document) => document.id === sourceDocumentId)) {
        setSelectedDocumentId(sourceDocumentId);
      }
      setMessages((data.messages || []).map((message) => ({ ...message, id: `saved-${message.id}` })));
    } catch (requestError) { setError(requestError.message); }
  }

  function newConversation() { setSessionId(""); setMessages([]); setQuestion(""); }

  async function sendQuestion() {
    const prompt = question.trim();
    if (!prompt || asking) return;
    if (!selectedDocumentId) return setError("Upload or select a completed document first.");
    setQuestion(""); setError(""); setAsking(true);
    setMessages((current) => [...current, { id: `user-${Date.now()}`, role: "user", content: prompt }]);
    try {
      const data = await askQuestion({ prompt, document_id: selectedDocumentId, session_id: sessionId || null, temperature: 0.3, top_p: 0.9 });
      if (data.session_id) setSessionId(data.session_id);
      setMessages((current) => [...current, { id: `assistant-${Date.now()}`, role: "assistant", content: data.answer, sources: data.sources || [] }]);
      const conversationData = await listConversations(); setConversations(conversationData.conversations || []);
    } catch (requestError) { setError(requestError.message); }
    finally { setAsking(false); }
  }

  function handleQuestionKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuestion(); }
  }

  if (authLoading) return <main className="auth-shell"><div className="spinner large" /></main>;
  if (!session) return <main className="auth-shell"><section className="auth-card">
    <div className="brand-mark"><Icon>◆</Icon></div><p className="eyebrow">BIZINTEL KNOWLEDGE</p>
    <h1>{authMode === "register" ? "Create your account." : "Your documents, ready to answer."}</h1>
    <p className="muted">Private document search secured by your Supabase account.</p>
    <form onSubmit={submitAuth} className="auth-form">
      <label>Email<input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@company.com" /></label>
      <label>Password<input type="password" autoComplete={authMode === "register" ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="••••••••" /></label>
      <button className="primary-button" disabled={loggingIn} type="submit">{loggingIn ? "Please wait" : authMode === "register" ? "Create account" : "Sign in"}</button>
    </form>
    <div className="auth-actions"><button type="button" onClick={() => { setAuthMode(authMode === "login" ? "register" : "login"); setLoginError(""); setAuthMessage(""); }}>{authMode === "login" ? "Create account" : "Back to sign in"}</button>{authMode === "login" && <button type="button" onClick={recoverPassword}>Forgot password?</button>}</div>
    {loginError && <p className="error-banner">{loginError}</p>}{authMessage && <p className="success-banner">{authMessage}</p>}
  </section></main>;

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark small"><Icon>◆</Icon></div><div><strong>BizIntel</strong><span>Document intelligence</span></div></div>
      <button className="upload-button" onClick={() => fileInputRef.current?.click()} disabled={uploading}>{uploading ? "Processing" : <><Icon>＋</Icon> Upload document</>}</button>
      <input ref={fileInputRef} className="visually-hidden" type="file" accept={ACCEPTED_FILES} onChange={handleFileSelect} /><p className="upload-help">Select a document before asking a question.</p>
      <section className="document-list" aria-label="Uploaded documents"><div className="section-heading"><span>Documents</span><span>{documents.length}</span></div>
        {documents.length === 0 ? <p className="empty-list">No uploaded documents.</p> : documents.map((document) => <article className={`document-item ${selectedDocumentId === document.id ? "selected" : ""}`} key={document.id}>
          <button className="document-select" onClick={() => setSelectedDocumentId(document.id)}><Icon>▤</Icon><div><strong title={document.filename}>{document.filename}</strong><span>{document.filename.split(".").pop()?.toUpperCase()} · {document.status} · {document.chunk_count} chunks</span><time>{new Date(document.uploaded_at).toLocaleString()}</time></div></button>
          <button className="delete-button" onClick={() => removeDocument(document.id)} title="Delete document" aria-label={`Delete ${document.filename}`}>×</button>
        </article>)}
      </section>
      <section className="conversation-list" aria-label="Conversation history"><div className="section-heading"><span>Conversations</span><button onClick={newConversation}>＋ New</button></div>
        {conversations.length === 0 ? <p className="empty-list">No conversations yet.</p> : conversations.map((conversation) => <button className={`conversation-item ${sessionId === conversation.session_id ? "selected" : ""}`} key={conversation.id} onClick={() => openConversation(conversation)}><strong>{conversation.title || "New conversation"}</strong><time>{new Date(conversation.created_at).toLocaleString()}</time></button>)}
      </section>
      <div className="account-card"><div className="avatar">{(session.user.email?.[0] || "U").toUpperCase()}</div><div><strong>{session.user.email}</strong><span>Authenticated</span></div><button onClick={logout} title="Log out">↗</button></div>
    </aside>
    <section className="chat-panel"><header className="chat-header"><div><p className="eyebrow">PRIVATE WORKSPACE</p><h1>Ask your documents</h1></div><span className="secure-badge"><Icon>●</Icon> User-isolated retrieval</span></header>
      <div className="messages" aria-live="polite">{messages.length === 0 ? <section className="welcome-state"><div className="welcome-icon"><Icon>✦</Icon></div><h2>Turn documents into answers</h2><p>Select a document, then ask a specific question. Answers use only retrieved context from that file.</p></section> : messages.map((message) => <article className={`message-row ${message.role}`} key={message.id}><div className="message-avatar">{message.role === "user" ? (session.user.email?.[0] || "U").toUpperCase() : "◆"}</div><div className="message-content"><span>{message.role === "user" ? "You" : "BizIntel"}</span><p>{message.content}</p>{message.sources?.length > 0 && <details><summary>{message.sources.length} source{message.sources.length === 1 ? "" : "s"}</summary><div className="sources">{message.sources.map((source, index) => <div key={`${source.document_id || source.source || "source"}-${index}`}><strong>{source.source || "Uploaded document"}</strong>{source.text && <p>{source.text}</p>}</div>)}</div></details>}</div></article>)}{asking && <article className="message-row assistant"><div className="message-avatar">◆</div><div className="message-content"><span>BizIntel</span><div className="typing"><i /><i /><i /></div></div></article>}<div ref={messagesEndRef} /></div>
      <footer className="composer-area">{error && <div className="error-banner"><Icon>!</Icon><span>{error}</span><button onClick={() => setError("")}>×</button></div>}<div className="composer"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleQuestionKeyDown} disabled={asking} rows={2} maxLength={4000} placeholder={selectedDocumentId ? "Ask about the selected document…" : "Upload or select a document first…"} /><button className="send-button" onClick={sendQuestion} disabled={asking || !question.trim() || !selectedDocumentId}>↑</button></div><p>Enter to send · Shift + Enter for a new line</p></footer>
    </section>
  </main>;
}

export default App;
