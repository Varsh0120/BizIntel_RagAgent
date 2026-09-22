import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, Loader, AlertCircle, Check, FileText } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  chunks?: Chunk[];
}

interface Chunk {
  text: string;
  source: string;
  file_type: string;
  section?: string;
}

interface UploadedFile {
  name: string;
  chunks: number;
  documentId: string;
}

export default function RAGChatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
  const ANTHROPIC_API_KEY = process.env.REACT_APP_ANTHROPIC_API_KEY || '';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const uploadDocument = async (file: File) => {
    if (!file.type.includes('pdf') && !file.type.includes('docx')) {
      setError('Only PDF and DOCX files are supported');
      return;
    }

    setUploadLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${BACKEND_URL}/upload-document`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();

      setUploadedFiles([
        ...uploadedFiles,
        {
          name: data.uploaded_file_name,
          chunks: data.chunks_stored,
          documentId: data.document_id,
        },
      ]);

      setMessages([
        ...messages,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: `✅ Document "${data.uploaded_file_name}" uploaded successfully with ${data.chunks_stored} chunks stored.`,
        },
      ]);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMsg);
      setMessages([
        ...messages,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: `❌ Upload failed: ${errorMsg}`,
        },
      ]);
    } finally {
      setUploadLoading(false);
    }
  };

  const retrieveChunks = async (query: string) => {
    try {
      const response = await fetch(`${BACKEND_URL}/ask-simple`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Failed to retrieve chunks');
      }

      const data = await response.json();
      return data.chunks as Chunk[];
    } catch (err) {
      console.error('Retrieval error:', err);
      return [];
    }
  };

  const generateAnswer = async (query: string, chunks: Chunk[]) => {
    if (!ANTHROPIC_API_KEY) {
      throw new Error('Anthropic API key not configured');
    }

    if (chunks.length === 0) {
      return 'No relevant information found in the uploaded documents.';
    }

    const context = chunks
      .map((chunk) => `Source: ${chunk.source}\n${chunk.text}`)
      .join('\n\n---\n\n');

    const systemPrompt = `You are a helpful assistant that answers questions based on provided document context.

STRICT RULES:
1. Answer only using the provided document context.
2. Do not invent or hallucinate information.
3. If the context does not contain the answer, say "I don't have information about that in the provided documents."
4. Be concise and factual.
5. Cite the source document when relevant.`;

    const conversationHistory = messages
      .filter((m) => m.role !== 'assistant' || !m.content.startsWith('❌'))
      .map((m) => ({
        role: m.role,
        content: m.content,
      }));

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        system: systemPrompt,
        messages: [
          ...conversationHistory,
          {
            role: 'user',
            content: `Context from documents:\n\n${context}\n\nQuestion: ${query}`,
          },
        ],
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || 'Failed to generate answer');
    }

    const data = await response.json();
    return data.content[0].text;
  };

  const handleSendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };

    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const chunks = await retrieveChunks(input);
      const answer = await generateAnswer(input, chunks);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: answer,
        chunks,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error generating answer';
      setError(errorMsg);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${errorMsg}`,
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadDocument(file);
      e.target.value = '';
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Sidebar */}
      <div className="w-64 border-r border-slate-700 bg-slate-800/50 p-4 flex flex-col">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <FileText className="w-6 h-6 text-blue-400" />
            RAG Chat
          </h1>
        </div>

        {/* Upload Section */}
        <div className="mb-8">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadLoading}
            className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
          >
            {uploadLoading ? (
              <>
                <Loader className="w-4 h-4 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Upload Document
              </>
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx"
            onChange={handleFileSelect}
            className="hidden"
          />
          <p className="text-xs text-slate-400 mt-2">PDF or DOCX files</p>
        </div>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Documents</h3>
            <div className="space-y-2">
              {uploadedFiles.map((file, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-slate-700/50 rounded-lg border border-slate-600"
                >
                  <p className="text-sm font-medium text-white truncate">{file.name}</p>
                  <p className="text-xs text-slate-400 mt-1">
                    <Check className="w-3 h-3 inline mr-1 text-green-400" />
                    {file.chunks} chunks
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Clear Button */}
        {messages.length > 0 && (
          <button
            onClick={() => {
              setMessages([]);
              setError(null);
            }}
            className="w-full mt-auto px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg font-medium transition-colors text-sm"
          >
            Clear Conversation
          </button>
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-slate-300 mb-2">
                  Start a Conversation
                </h2>
                <p className="text-slate-400">
                  Upload documents and ask questions about them
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-2xl rounded-lg p-4 ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-100'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>

                  {/* Show chunks for assistant messages */}
                  {message.chunks && message.chunks.length > 0 && (
                    <details className="mt-3 pt-3 border-t border-slate-600">
                      <summary className="cursor-pointer text-xs font-medium text-slate-300 hover:text-slate-200">
                        📄 {message.chunks.length} source(s)
                      </summary>
                      <div className="mt-2 space-y-2">
                        {message.chunks.map((chunk, idx) => (
                          <div
                            key={idx}
                            className="text-xs p-2 bg-slate-800/50 rounded border border-slate-600"
                          >
                            <p className="font-medium text-slate-300">
                              {chunk.source}
                              {chunk.section && ` - ${chunk.section}`}
                            </p>
                            <p className="text-slate-400 mt-1 line-clamp-3">
                              {chunk.text}
                            </p>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-700 rounded-lg p-4">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                  <div
                    className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                    style={{ animationDelay: '0.1s' }}
                  />
                  <div
                    className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                    style={{ animationDelay: '0.2s' }}
                  />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="mx-6 mb-4 p-3 bg-red-900/20 border border-red-700 rounded-lg flex gap-2 items-start">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {/* Input Area */}
        <div className="border-t border-slate-700 bg-slate-800/50 p-4">
          <div className="flex gap-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your documents..."
              disabled={loading}
              rows={3}
              className="flex-1 bg-slate-700 text-white placeholder-slate-400 rounded-lg p-3 resize-none border border-slate-600 focus:border-blue-500 focus:outline-none disabled:opacity-50 text-sm"
            />
            <button
              onClick={handleSendMessage}
              disabled={loading || !input.trim()}
              className="px-4 h-fit bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white rounded-lg font-medium flex items-center justify-center gap-2 transition-colors self-end"
            >
              {loading ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            Press Shift+Enter for new line, Enter to send
          </p>
        </div>
      </div>
    </div>
  );
}