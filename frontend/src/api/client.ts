/**
 * API-клиент для бэкенда Фармбазис ИИ-Техподдержки.
 */

const API_BASE = '/api'

export interface SuggestedTopic {
  title: string
  article_id: string
  score: number
  snippet: string
}

export interface ChatResponse {
  answer: string
  session_id: string
  confidence: number
  needs_escalation: boolean
  source_articles: string[]
  youtube_links: string[]
  has_images: boolean
  response_type: 'answer' | 'clarification'
  suggested_topics: SuggestedTopic[] | null
}

export interface EscalationResponse {
  escalation_id: string
  status: string
  message: string
  position_in_queue: number
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}

export interface EscalationDetail {
  escalation_id: string
  session_id: string
  status: string
  reason: string | null
  contact_info: string | null
  chat_history: ChatMessage[]
  created_at: string
  updated_at: string | null
  operator_notes: string | null
}

export interface HealthStatus {
  status: string
  version: string
  knowledge_base_ready: boolean
  total_articles: number
  total_chunks: number
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  // === Чат ===

  async sendMessage(message: string, sessionId?: string): Promise<ChatResponse> {
    const res = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        session_id: sessionId || null,
      }),
    })
    if (!res.ok) throw new Error(`API error: ${res.status}`)
    return res.json()
  }

  // === Эскалация ===

  async createEscalation(
    sessionId: string,
    reason?: string,
    contactInfo?: string
  ): Promise<EscalationResponse> {
    const res = await fetch(`${this.baseUrl}/escalation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        reason,
        contact_info: contactInfo,
      }),
    })
    if (!res.ok) throw new Error(`API error: ${res.status}`)
    return res.json()
  }

  // === Обратная связь ===

  async sendFeedback(
    sessionId: string,
    rating: number,
    messageIndex: number = 0,
    comment?: string
  ): Promise<void> {
    await fetch(`${this.baseUrl}/escalation/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        rating,
        message_index: messageIndex,
        comment,
      }),
    })
  }

  // === Оператор ===

  async operatorLogin(username: string, password: string): Promise<{ token: string; username: string }> {
    const res = await fetch(`${this.baseUrl}/operator/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!res.ok) throw new Error('Неверный логин или пароль')
    return res.json()
  }

  async getEscalations(token: string, status?: string): Promise<{
    escalations: EscalationDetail[]
    total: number
    pending_count: number
  }> {
    const url = new URL(`${window.location.origin}${this.baseUrl}/operator/escalations`)
    if (status) url.searchParams.set('status', status)

    const res = await fetch(url.toString(), {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!res.ok) throw new Error(`API error: ${res.status}`)
    return res.json()
  }

  async operatorReply(
    token: string,
    escalationId: string,
    message: string,
    closeTicket: boolean = false
  ): Promise<void> {
    const res = await fetch(`${this.baseUrl}/operator/reply`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        escalation_id: escalationId,
        message,
        close_ticket: closeTicket,
      }),
    })
    if (!res.ok) throw new Error(`API error: ${res.status}`)
  }

  // === Система ===

  async healthCheck(): Promise<HealthStatus> {
    const res = await fetch(`${this.baseUrl}/health`)
    return res.json()
  }
}

export const api = new ApiClient()
